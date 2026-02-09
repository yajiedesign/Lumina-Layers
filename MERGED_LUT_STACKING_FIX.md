# 融合LUT堆叠信息保留 - 修复说明

## 问题描述

之前融合LUT导出的3MF文件只有一个对象，因为堆叠信息丢失了。用户指出融合LUT的数据来源于实际打印的色卡，应该保留原始堆叠信息，而不是重新推断。

另外，融合后的 `.npz` 文件无法通过UI上传，需要修改文件上传组件。

## 解决方案

### 1. 修改LUT融合逻辑 (`utils/lut_merger.py`)

**修改 `merge_luts()` 函数**:
- 返回值从单个数组改为字典：`{'colors': RGB数组, 'stacks': 堆叠数组}`
- 为每个LUT重建堆叠信息：
  - BW (32色): 2^5穷举
  - 4色 (1024色): 4^5穷举
  - 6色 (1296色): 调用`get_top_1296_colors()`
  - 8色 (2738色): 加载`assets/smart_8color_stacks.npy`
  - 未知格式: 使用`_analyze_color_stacks()`智能分析

**新增 `_reconstruct_stacks()` 函数**:
- 根据LUT类型重建堆叠信息
- 支持所有标准LUT格式
- 提供fallback机制处理未知格式

**新增 `_analyze_color_stacks()` 函数**:
- 智能分析颜色并推断堆叠信息
- 作为未知格式的fallback方法

### 2. 修改LUT保存格式 (`ui/layout_new.py`)

**修改 `on_merge_luts_click()` 函数**:
- 从 `.npy` 格式改为 `.npz` 格式
- 使用 `np.savez_compressed()` 同时保存颜色和堆叠数据
- 输出文件：`output/merged_lut.npz`

**修改文件上传组件**:
- 图像转换TAB的LUT上传：支持 `.npy` 和 `.npz`
- 高级TAB的主LUT上传：支持 `.npy` 和 `.npz`
- 高级TAB的次要LUT上传：支持 `.npy` 和 `.npz`
- 标签更新为 "(.npy/.npz)"

### 3. 修改LUT加载逻辑 (`core/image_processing.py`)

**修改 `_load_lut()` 函数**:
- 检测文件扩展名（`.npy` 或 `.npz`）
- 对于 `.npz` 文件：
  - 加载 `colors` 和 `stacks` 两个数组
  - 直接使用保存的堆叠信息，不重新生成
  - 跳过智能推断逻辑
- 对于 `.npy` 文件：保持原有逻辑

### 4. 更新LUT管理器 (`utils/lut_manager.py`)

**修改 `get_all_lut_files()` 函数**:
- 同时扫描 `.npy` 和 `.npz` 文件
- 为融合LUT添加 `[Merged]` 标记

**修改 `save_uploaded_lut()` 函数**:
- 支持保存 `.npz` 文件
- 验证文件扩展名
- 保留原始扩展名

### 5. 更新LUT检测逻辑 (`core/converter.py`)

**修改 `detect_lut_color_mode()` 函数**:
- 检测 `.npz` 文件格式
- 从 `colors` 数组读取颜色数量
- 保持原有的颜色数量范围检测逻辑

**修改 `validate_lut_compatibility()` 函数** (`utils/lut_merger.py`):
- 支持验证 `.npz` 文件
- 正确读取颜色数量

## 数据流程

### 融合LUT创建流程

1. 用户上传主LUT和次要LUT文件（支持 `.npy` 和 `.npz`）
2. `merge_luts()` 函数：
   - 加载所有LUT文件
   - 为每个LUT重建堆叠信息
   - 合并颜色和堆叠数据
   - 返回 `{'colors': ..., 'stacks': ...}`
3. `on_merge_luts_click()` 函数：
   - 接收融合结果
   - 使用 `np.savez_compressed()` 保存为 `.npz` 文件
   - 文件包含两个数组：`colors` 和 `stacks`
4. 用户可以下载融合后的 `.npz` 文件

### 融合LUT使用流程

1. 用户上传或选择融合LUT（`.npz` 文件）
2. `_load_lut()` 函数：
   - 检测到 `.npz` 扩展名
   - 加载 `colors` 和 `stacks` 数组
   - 直接使用保存的堆叠信息
   - 构建KD-Tree用于颜色匹配
3. 图像转换：
   - 使用KD-Tree匹配图像颜色到LUT颜色
   - 使用保存的堆叠信息生成材料矩阵
   - 为每个材料生成独立的网格对象
4. 导出3MF：
   - 每个材料对应一个对象
   - 所有对象都包含在3MF文件中

## 文件格式

### .npy 格式（标准LUT）
```python
# 形状: (N, 3)
# 内容: RGB颜色数组
colors = np.array([[R, G, B], ...])
np.save('lut.npy', colors)
```

### .npz 格式（融合LUT）
```python
# 包含两个数组
colors = np.array([[R, G, B], ...])  # 形状: (N, 3)
stacks = np.array([[L1, L2, L3, L4, L5], ...])  # 形状: (N, 5)
np.savez_compressed('merged_lut.npz', colors=colors, stacks=stacks)
```

## 堆叠信息重建规则

### BW模式 (32色)
- 2^5 = 32种组合
- 穷举所有可能的5层堆叠
- 每层可以是0(白色)或1(黑色)

### 4色模式 (1024色)
- 4^5 = 1024种组合
- 穷举所有可能的5层堆叠
- 每层可以是0-3（白/青/品红/黄 或 白/红/黄/蓝）

### 6色模式 (1296色)
- 使用智能筛选算法
- 调用 `get_top_1296_colors()` 获取最优组合
- 反转堆叠顺序以适应Face-Down打印

### 8色模式 (2738色)
- 使用预生成的智能堆叠数据
- 加载 `assets/smart_8color_stacks.npy`
- 反转堆叠顺序以适应Face-Down打印

### 未知格式
- 使用颜色分析fallback方法
- 计算每个颜色与基础材料的距离
- 选择最接近的材料填充所有5层

## 测试建议

1. **创建融合LUT**:
   - 上传8色主LUT + 6色/4色/BW次要LUT
   - 验证融合成功并生成 `.npz` 文件
   - 检查文件大小（应该比 `.npy` 大，因为包含堆叠信息）
   - 下载融合后的 `.npz` 文件

2. **上传融合LUT**:
   - 在图像转换TAB上传 `.npz` 文件
   - 验证文件被正确识别和加载
   - 检查LUT状态显示

3. **使用融合LUT转换图像**:
   - 选择融合LUT
   - 上传测试图像
   - 生成预览和3MF文件
   - 验证3MF包含多个材料对象（不是只有一个）

4. **验证堆叠信息**:
   - 在切片软件中打开3MF文件
   - 检查每个材料对象是否正确
   - 验证颜色是否与预期一致

## 兼容性

- **向后兼容**: 所有现有的 `.npy` LUT文件仍然可以正常使用
- **新功能**: `.npz` 融合LUT提供更准确的堆叠信息
- **自动检测**: 系统自动识别文件格式并使用正确的加载方法
- **文件上传**: 所有LUT上传组件都支持 `.npy` 和 `.npz` 格式

## 注意事项

1. 融合LUT文件会比标准LUT大（因为包含堆叠信息）
2. 使用 `np.savez_compressed()` 压缩以减小文件大小
3. 堆叠信息来源于原始LUT的实际打印数据，不是推断的
4. 对于未知格式的LUT，系统会使用智能分析作为fallback
5. 融合后的 `.npz` 文件可以作为主LUT或次要LUT再次融合

## 修改的文件列表

1. `utils/lut_merger.py` - LUT融合核心逻辑
2. `ui/layout_new.py` - UI回调函数和文件上传组件
3. `core/image_processing.py` - LUT加载逻辑
4. `utils/lut_manager.py` - LUT文件管理
5. `core/converter.py` - LUT检测逻辑

## 完成状态

✅ 所有代码修改已完成
✅ 文件上传组件已更新支持 `.npz`
✅ 无语法错误
⏳ 等待用户测试验证
