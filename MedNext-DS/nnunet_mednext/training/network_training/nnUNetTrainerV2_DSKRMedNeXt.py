import torch

from nnunet_mednext.utilities.nd_softmax import softmax_helper
from nnunet_mednext.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import \
    nnUNetTrainerV2_MedNeXt_S_kernel3

from nnunet_mednext.network_architecture.MSRSDMedNeXt import MSRSDMedNeXt
from nnunet_mednext.training.loss_functions.msrsd_loss import MSRSDLoss


class nnUNetTrainerV2_MSRSDMedNeXt(nnUNetTrainerV2_MedNeXt_S_kernel3):
    """
    保留图中 Teacher/Student skeleton，同时用 RSDV2 的高分辨率 residual refinement
    """

    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16=False
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16
        )

        self.max_num_epochs = 520
        self.initial_lr = 2e-4
        self.weight_decay = 1e-2
        self.pin_memory = True

        self._last_stage = None
        self.stage0_end = 40
        self.stage1_end = 120
        self.stage2_end = 260

    def initialize_network(self):
        self.network = MSRSDMedNeXt(
            input_channels=self.num_input_channels,
            num_classes=self.num_classes,
            base_channels=24,
            refine_channels=16
        )

        if torch.cuda.is_available():
            self.network.cuda()

        self.network.inference_apply_nonlin = softmax_helper

    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)

        self.loss = MSRSDLoss(
            soft_dice_kwargs={
                'batch_dice': self.batch_dice,
                'smooth': 1e-5,
                'do_bg': False
            },
            ce_kwargs={},
            aggregate="sum",
            coarse_weight=0.28,
            edge_weight=0.15,
            teacher_weight=0.08,
            student_weight=0.10,
            et_weight=0.40,
            delta_bg_weight=0.01
        )

        self._apply_training_stage(force=True)

    @staticmethod
    def _set_module_trainable(module, flag: bool):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = flag

    def _set_by_name(self, module_name, flag: bool):
        if hasattr(self.network, module_name):
            self._set_module_trainable(getattr(self.network, module_name), flag)

    def _apply_training_stage(self, force=False):
        ep = int(self.epoch)

        if ep < self.stage0_end:
            stage = 0
        elif ep < self.stage1_end:
            stage = 1
        elif ep < self.stage2_end:
            stage = 2
        else:
            stage = 3

        if (not force) and (stage == self._last_stage):
            return
        self._last_stage = stage

        # 默认全开
        for name in [
            "teacher_sba", "teacher_aux",
            "student_sda", "student_rag", "student_edgebikan", "student_aux",
            "dec_rag_d2", "dec_rag_d1",
            "refine_proj", "refine_pre", "gate",
            "context_detail", "edge_detail", "fuse_weight",
            "edge_head"
        ]:
            self._set_by_name(name, True)

        if stage == 0:
            # 先稳 coarse + teacher
            for name in [
                "student_sda", "student_rag", "student_edgebikan", "student_aux",
                "refine_proj", "refine_pre", "gate",
                "context_detail", "edge_detail", "fuse_weight"
            ]:
                self._set_by_name(name, False)

            self.loss.coarse_weight = 0.42
            self.loss.edge_weight = 0.04
            self.loss.teacher_weight = 0.08
            self.loss.student_weight = 0.02
            self.loss.et_weight = 0.10
            self.loss.delta_bg_weight = 0.0
            print("[MSRSD Stage 0] coarse + teacher stabilization", flush=True)

        elif stage == 1:
            # 打开 student path 和 refine_pre，但 gate 先冻结
            for name in [
                "student_sda", "student_rag", "student_edgebikan", "student_aux",
                "refine_proj", "refine_pre",
                "context_detail", "edge_detail", "fuse_weight"
            ]:
                self._set_by_name(name, True)
            self._set_by_name("gate", False)

            self.loss.coarse_weight = 0.34
            self.loss.edge_weight = 0.08
            self.loss.teacher_weight = 0.06
            self.loss.student_weight = 0.08
            self.loss.et_weight = 0.22
            self.loss.delta_bg_weight = 0.005
            print("[MSRSD Stage 1] student/refine on, gate frozen", flush=True)

        elif stage == 2:
            # 全部打开
            self.loss.coarse_weight = 0.26
            self.loss.edge_weight = 0.14
            self.loss.teacher_weight = 0.04
            self.loss.student_weight = 0.08
            self.loss.et_weight = 0.34
            self.loss.delta_bg_weight = 0.01
            print("[MSRSD Stage 2] full refinement", flush=True)

        else:
            # 后期强调 ET 和边界
            self.loss.coarse_weight = 0.20
            self.loss.edge_weight = 0.18
            self.loss.teacher_weight = 0.03
            self.loss.student_weight = 0.10
            self.loss.et_weight = 0.44
            self.loss.delta_bg_weight = 0.015
            print("[MSRSD Stage 3] ET/boundary emphasized", flush=True)

    @staticmethod
    def _unwrap_target_for_eval(target):
        if isinstance(target, (list, tuple)):
            target = target[0]
        if target.ndim == 5:
            target = target[:, 0]
        return target.long()

    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            target = self._unwrap_target_for_eval(target)
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)

            num_classes = output.shape[1]
            axes = tuple(range(1, target.ndim))

            tp_hard = torch.zeros((target.shape[0], num_classes - 1), device=output.device)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1), device=output.device)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1), device=output.device)

            for c in range(1, num_classes):
                pred_c = (output_seg == c).float()
                target_c = (target == c).float()

                tp_hard[:, c - 1] = (pred_c * target_c).sum(dim=axes)
                fp_hard[:, c - 1] = (pred_c * (1 - target_c)).sum(dim=axes)
                fn_hard[:, c - 1] = ((1 - pred_c) * target_c).sum(dim=axes)

            tp_hard = tp_hard.sum(0).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0).detach().cpu().numpy()

            dc = (2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)

            self.online_eval_foreground_dc.append(list(dc))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        self._apply_training_stage()

        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad(set_to_none=True)

        if self.fp16:
            with torch.amp.autocast(device_type='cuda', enabled=True):
                outputs = self.network.forward_all(data)
                loss = self.loss(outputs, target)

            if not torch.isfinite(loss):
                print("[MSRSD Warning] non-finite loss in fp16 path, skip step", flush=True)
                return 0.0

            if do_backprop:
                self.amp_grad_scaler.scale(loss).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 8.0)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            outputs = self.network.forward_all(data)
            loss = self.loss(outputs, target)

            if not torch.isfinite(loss):
                print("[MSRSD Warning] non-finite loss in fp32 path, skip step", flush=True)
                return 0.0

            if do_backprop:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 8.0)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(outputs["seg"], target)

        del target
        return loss.detach().cpu().numpy()