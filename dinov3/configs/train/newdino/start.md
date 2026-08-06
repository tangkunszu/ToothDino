# newdino 启动命令

## 必读：不要用 `conda activate dinov3`

那个环境里 `import dinov3` 解析到的是**另一个旧仓库** `/data/tangkun/project/dinov3/`，
本目录所有配置依赖的改动（`n_tcc`、`tcc_legacy`、`blur_probability_*`、n-TCC 的 H² 锚定、
`global_crops_ratio`、`teacher_no_color_jitter`、解码上限、band 缓存）在那边**一个都没有**。

用它启动会静默跑旧代码 —— `local_crop_strategy: n_tcc` 落到 `else` 分支变成官方随机裁剪，
**而且不报错**。你会拿到一个看起来正常、实际什么改动都没有的 run。

正确做法是显式设 `PYTHONPATH`（`torchrun` 把**脚本所在目录**而非当前目录放进 `sys.path`，
不设就会 `ModuleNotFoundError: No module named 'dinov3'`）。

```bash
export REPO=/data/tangkun/project/dinov3newimprove/dinov3
export PY=/data/tangkun/anaconda3/envs/dinov3/bin/torchrun
```

---

## 阶段一：主训练

```bash
cd $REPO
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$REPO $PY \
  --standalone --nproc_per_node=4 \
  dinov3/train/train.py \
  --config-file dinov3/configs/train/newdino/vitb_plus_davc_abm.yaml \
  --output-dir $REPO/output/newdino/vitb_plus_davc_abm
```

消融链的其余三个换掉 config 与 output-dir 即可：

```
vitb_baseline.yaml        Baseline
vitb_plus_davc.yaml       + DAVC
vitb_plus_abm.yaml        + ABM
vitb_plus_davc_abm.yaml   + DAVC + ABM（主结果）
```

主链外的两个单变量变体（各只与 `vitb_plus_davc_abm.yaml` 差一行）：

```
var_davc_abm_wideview.yaml      + global_crops_ratio: [1.2, 2.6]
var_davc_abm_cleanteacher.yaml  + teacher_no_color_jitter: true
```

---

## 阶段二：高分辨率适配（主训练结束后）

先把阶段一产出的 teacher checkpoint 填进配置的两处 `SET_ME_...`：

```bash
CKPT=$REPO/output/newdino/vitb_plus_davc_abm/eval/training_<N>/teacher_checkpoint.pth
sed -i "s|SET_ME_to_stage1_teacher_checkpoint.pth|$CKPT|g" \
  dinov3/configs/train/newdino/stage2_high_res_adapt.yaml
```

```bash
cd $REPO
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$REPO $PY \
  --standalone --nproc_per_node=4 \
  dinov3/train/train.py \
  --config-file dinov3/configs/train/newdino/stage2_high_res_adapt.yaml \
  --output-dir $REPO/output/newdino/stage2_high_res_adapt
```

阶段二的超参是从官方 7B 配方缩放来的（batch 64→16、单一 512 分辨率、去掉 Gram），
**未经验证**。盯前几百步，loss 不收敛就调低 `optim.lr`。

---

## 启动后确认（很重要）

```bash
L=$REPO/output/newdino/launch_davc_abm.log
grep -m1 "Student checkpoint loaded" $L | grep -o "unexpected_keys=\[\]"   # 官方权重已载入
grep -m1 "local_crop_strategy:"       $L                                   # 应为 n_tcc
grep -m1 "blur probability"           $L                                   # 应为 0.30/0.10/0.40
grep -m1 "representative_tooth_cache" $L                                   # 缓存路径
grep -m1 "global_crops_ratio"         $L                                   # 变体才有
```

曾经踩过的坑：`blur_probability_global2` 和 `blur_probability_local` 在 `ssl_meta_arch.py`
里漏了接线，config 写 0.10 实际跑 0.15，只有日志能看出来 —— **每次开跑都对一遍这几行**。

---

## eval checkpoint 快照

`evaluation.eval_checkpoint_max_to_keep` 只在启动时读取，改配置对已在跑的 run 无效。
需要中间 checkpoint（用 `diagnose_patch_correspondence.py` 测稠密特征是否随训练退化，
以判断 DINOv3 的 Gram anchoring 前提是否适用于 44.6k 步的规模）时，用守护进程在被覆盖前抄走：

```bash
cd $REPO && nohup ./snapshot_eval_checkpoints.sh > output/newdino/snapshot_watch.log 2>&1 &
```

默认最多 5 个、剩余空间低于 40G 自动停止。

---

## 数据

预训练池：`/data/tangkun/pXray/All/train/pXray/`，**51,654 张**

- 已移出 8,457 张 Roboflow 预增强副本 → `All/train/duplicate/roboflow_dups/`
  （每个源 study 保留 1 张，`roboflow_dedup_manifest.json` 可完整还原）
- 已并入 2,879 张 `pXray/xray/` 临床全景片（零内部重复、与下游评测零重合）
- band 缓存：`/data/tangkun/pXray/All/train/tooth_center_cache.json`（51,654 条）
  **换数据池后必须重建**：
  ```bash
  python build_tooth_center_cache.py --root <pool> --out <cache.json> --workers 24
  ```
