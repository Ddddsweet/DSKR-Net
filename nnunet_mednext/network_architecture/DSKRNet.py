import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper
from nnunet_mednext.network_architecture.msrsd_modules import (
    ConvNormAct, DWConvBlock, DownBlock, UpBlock,
    SBA as StructurePriorAdapter,
    SDA as SelectiveDetailCalibration,
    RAG as StructureGuidedAggregation,
    EdgeBiKAN as KANSplineReparameterization,
    RegionPriorHead as CoarsePredictionHead,
    StructureEdgeHead as BoundaryCueHead,
    UncertaintyHead,
    TeacherAuxHead as PriorAuxHead,
    StudentAuxHead as DetailAuxHead,
    HybridETGate as StructureConstrainedUpdateGate,
    ContextDetailBranch as ContextPreservingBranch,
    EdgeDetailBranch as EdgeSensitiveBranch,
    AdaptiveFuseHead as AdaptiveCorrectionFuseHead,
)


class DSKRNet(SegmentationNetwork):
    """
    DSKRNet: Detail Selective KAN Reparameterization Network with Structural Prior.

    This implementation keeps the original computational graph unchanged and only
    renames modules/variables to match the paper terminology:

        Input -> Stem -> Encoder -> SPA-related structural guidance
              -> Decoder -> SDKR-related local correction
              -> SCU-style conservative residual write-back -> Output

    Paper-level naming used in this file:
        SPA-related path:
            structure_prior_adapter, prior_aux_head
        SDKR-related path:
            selective_detail_calibration, kan_reparameterization,
            context_preserving_branch, edge_sensitive_branch,
            adaptive_correction_fuse
        SCU-related path:
            coarse_prediction_head, boundary_cue_head, uncertainty_head,
            structure_update_gate

    Only module and variable names are aligned with the paper terminology. The layer order, forward computation, tensor operations, and returned segmentation logits remain unchanged.
    """

    def __init__(self, input_channels=4, num_classes=4, base_channels=24, refine_channels=16):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.num_classes = num_classes
        self._deep_supervision = False
        self.do_ds = False
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = np.array([16, 16, 16])

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 10
        cr = refine_channels

        # Stem
        self.stem = nn.Sequential(
            ConvNormAct(input_channels, c1, 3, 1, 1),
            ConvNormAct(c1, c1, 1, 1, 0),
        )

        # Encoder
        self.enc1 = nn.Sequential(DWConvBlock(c1), DWConvBlock(c1))
        self.down1 = DownBlock(c1, c2)

        self.enc2 = nn.Sequential(DWConvBlock(c2), DWConvBlock(c2))
        self.down2 = DownBlock(c2, c3)

        self.enc3 = nn.Sequential(DWConvBlock(c3), DWConvBlock(c3))
        self.down3 = DownBlock(c3, c4)

        self.enc4 = nn.Sequential(DWConvBlock(c4), DWConvBlock(c4))
        self.down4 = DownBlock(c4, c5)

        # SPA-related structural guidance branch.
        # Original name: teacher_sba / teacher_aux.
        self.structure_prior_adapter = StructurePriorAdapter(c3)
        self.prior_aux_head = PriorAuxHead(c3)

        # SDKR-related selective local correction branch.
        # Original name: student_sda / student_rag / student_edgebikan / student_aux.
        self.selective_detail_calibration = SelectiveDetailCalibration(c3)
        self.structure_guided_detail_aggregation = StructureGuidedAggregation(c3, c3)
        self.kan_reparameterization = KANSplineReparameterization(c3)
        self.detail_aux_head = DetailAuxHead(c3)

        # Bottleneck with structure-guided aggregation.
        self.bottleneck = nn.Sequential(DWConvBlock(c5), DWConvBlock(c5))
        self.bottleneck_structure_aggregation = StructureGuidedAggregation(c5, c3)

        # Decoder
        self.up4 = UpBlock(c5, c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.decoder_structure_aggregation_d2 = StructureGuidedAggregation(c2, c3)
        self.decoder_structure_aggregation_d1 = StructureGuidedAggregation(c1, c3)

        # SCU-style conservative residual write-back heads.
        # These names match the paper-level coarse prediction, boundary/uncertainty cues,
        # gated update, and dual correction branch descriptions.
        self.coarse_prediction_head = CoarsePredictionHead(c2, num_classes)
        self.boundary_cue_head = BoundaryCueHead(c2)
        self.uncertainty_head = UncertaintyHead()

        self.refine_projection = ConvNormAct(c1, cr, 1, 1, 0)
        self.refine_preparation = nn.Sequential(
            ConvNormAct(cr + num_classes + 1 + 1 + 1 + 1, cr, 3, 1, 1),
            DWConvBlock(cr),
        )

        self.structure_update_gate = StructureConstrainedUpdateGate(cr, num_classes)
        self.context_preserving_branch = ContextPreservingBranch(cr, num_classes)
        self.edge_sensitive_branch = EdgeSensitiveBranch(cr, num_classes)
        self.adaptive_correction_fuse = AdaptiveCorrectionFuseHead(cr)

    def forward_features(self, x):
        x1 = self.enc1(self.stem(x))
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        x4 = self.enc4(self.down3(x3))
        xb = self.bottleneck(self.down4(x4))
        return x1, x2, x3, x4, xb

    def forward_all(self, x):
        x1, x2, x3, x4, xb = self.forward_features(x)

        # SPA-related structural guidance.
        structural_prior_feature = self.structure_prior_adapter(x3)
        structural_prior_map = self.prior_aux_head(structural_prior_feature)

        # SDKR-related selective detail correction.
        detail_seed = self.selective_detail_calibration(x3)
        local_correction_feature = self.structure_guided_detail_aggregation(
            detail_seed, structural_prior_feature
        )
        local_correction_feature = self.kan_reparameterization(local_correction_feature)
        local_correction_map = self.detail_aux_head(local_correction_feature)

        # Bottleneck with structure-guided aggregation.
        xb = self.bottleneck_structure_aggregation(xb, local_correction_feature)

        # Decoder.
        d4 = self.up4(xb, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d2 = self.decoder_structure_aggregation_d2(d2, local_correction_feature)

        d1 = self.up1(d2, x1)
        d1 = self.decoder_structure_aggregation_d1(d1, structural_prior_feature)

        # Coarse prediction and geometry/uncertainty cues.
        coarse_low = self.coarse_prediction_head(d2)
        boundary_low = self.boundary_cue_head(d2)
        uncertainty_low = self.uncertainty_head(coarse_low)

        coarse = F.interpolate(coarse_low, size=d1.shape[2:], mode="trilinear", align_corners=False)
        boundary = F.interpolate(boundary_low, size=d1.shape[2:], mode="trilinear", align_corners=False)
        uncertainty = F.interpolate(uncertainty_low, size=d1.shape[2:], mode="trilinear", align_corners=False)

        structural_prior_map_up = F.interpolate(
            structural_prior_map, size=d1.shape[2:], mode="trilinear", align_corners=False
        )
        local_correction_map_up = F.interpolate(
            local_correction_map, size=d1.shape[2:], mode="trilinear", align_corners=False
        )

        prob = torch.softmax(coarse, dim=1)
        if coarse.shape[1] >= 4:
            et_prior = prob[:, 3:4]
        else:
            et_prior = prob[:, -1:]

        refine_base = self.refine_projection(d1)
        refine_in = torch.cat(
            [refine_base, coarse, uncertainty, boundary, structural_prior_map_up, local_correction_map_up],
            dim=1,
        )
        refine_feat = self.refine_preparation(refine_in)

        update_gate = self.structure_update_gate(
            refine_feat,
            coarse,
            uncertainty,
            boundary,
            et_prior,
            structural_prior_map_up,
            local_correction_map_up,
        )
        gated_feat = update_gate * refine_feat

        context_delta = self.context_preserving_branch(gated_feat)
        edge_delta = self.edge_sensitive_branch(gated_feat)

        fuse_w = self.adaptive_correction_fuse(gated_feat)
        delta = fuse_w[:, 0:1] * context_delta + fuse_w[:, 1:2] * edge_delta

        # Conservative residual write-back.
        final_logits = coarse + delta

        return {
            "seg": final_logits,
            "coarse": coarse_low,
            "edge": boundary,                      # kept for compatibility with existing code
            "boundary": boundary,                  # paper-consistent alias
            "uncertainty": uncertainty,
            "teacher_map": structural_prior_map,    # kept for compatibility with existing code
            "student_map": local_correction_map,    # kept for compatibility with existing code
            "structural_prior_map": structural_prior_map,
            "local_correction_map": local_correction_map,
            "delta": delta,
            "context_delta": context_delta,
            "edge_delta": edge_delta,
            "gate": update_gate,
            "update_gate": update_gate,
        }

    def forward(self, x):
        return self.forward_all(x)["seg"]
