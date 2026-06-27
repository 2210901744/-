# Learning Screen Assistant

一个用于个人学习自测的屏幕题目辅助工具原型。

## 重要免责声明

本项目仅面向**个人学习、自测、题库整理、软件功能验证**或你自己控制的练习系统使用。请勿将本工具用于正式考试、线上测评、招聘笔试、课程考核、竞赛、认证考试，或任何平台/学校/机构明确禁止使用辅助工具的场景。

使用者应自行确认使用场景是否合法合规，并自行承担因使用本工具造成的后果。本工具不会保证答案正确性，也不会保证 OCR、DeepSeek、自动点击和随机兜底结果稳定可靠。自动点击可能误点、漏点或跳题；随机兜底本质上是猜测，可能产生错误答案。请在明确了解风险后再启用相关功能。

本工具会把 OCR 识别到的题目文本发送给 DeepSeek 接口进行回答。请不要输入或截取包含个人隐私、账号密码、保密资料、考试原题库、商业机密或其他不应上传到第三方 API 的内容。

## 功能概览

当前版本已移除本地 SQLite 题库逻辑，所有题目均直接请求 DeepSeek，不再进行本地题库检索、导入、缓存、保存或删除。

主要功能：

- 截取当前屏幕或指定区域，并使用 OCR 识别题目和选项文本；
- 将识别到的题目文本发送给 DeepSeek，获取题型、答案、置信度和解释；
- 支持单选题、多选题和判断题；
- 支持框选题目区域，减少 OCR 范围，提高识别速度；
- 支持自动点击答案，默认关闭，仅在手动启用后执行；
- 支持多选题点击“保存/提交/确定”等按钮；
- 支持保存后等待页面稳定，再判断是否需要继续点击“下一题”；
- 支持点击后鼠标复位，避免鼠标遮挡下一题题干影响 OCR；
- 支持点击前双重坐标校验，坐标波动过大时进行第三次计算并取稳定中点；
- 支持选项字母定位失败后，使用选项正文文本进行模糊匹配；
- 支持多选题只识别出一个答案时，随机补选一个未选项；
- 支持识别失败时的随机兜底，尽量保证循环流程不中断；
- 支持循环模式：OCR 识别 → 请求 DeepSeek → 点击答案 → 必要时保存 → 必要时进入下一题。

## 文件说明

核心文件：

- `app.py`：主程序，单文件实现；
- `requirements.txt`：依赖列表；
- `.env.example`：DeepSeek 配置示例；
- `click_config.json`：点击配置文件，运行后自动生成或更新。

当前版本不再依赖以下旧版题库文件：

- `question_bank.sqlite3`；
- `sample_questions.csv`；
- `import_jinling_pdf_bank.py`；
- 任何由 PDF 或 CSV 生成的本地题库文件。

如果项目目录中仍保留这些旧文件，它们不会参与当前 DeepSeek-only 流程。

## 安装

建议使用 Python 3.10+。

PowerShell 示例：

```powershell
cd 你的项目目录
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

PaddleOCR 首次安装和初始化可能较慢。如果只想测试 DeepSeek 查询功能，可以暂时不使用 OCR，直接在窗口文本框中粘贴题目和选项后点击“请求 DeepSeek”。

## 配置 DeepSeek

复制配置文件：

```powershell
copy .env.example .env
```

然后编辑 `.env`：

```env
DEEPSEEK_API_KEY=你的_key
DEEPSEEK_MODEL=deepseek-chat
```

也可以直接设置系统环境变量。

没有配置 `DEEPSEEK_API_KEY` 时，程序可以启动，但无法完成答题查询。

## 运行

```powershell 需要在管理员权限下使用
python app.py
```

顶部按钮：

- `0 识别当前屏幕`：截屏并 OCR，把识别结果填入题目文本框；
- `1 请求 DeepSeek`：将当前题目文本框内容发送给 DeepSeek；
- `3 点击当前答案`：根据当前答案执行一次自动点击；
- `4 开始答题`：进入循环模式；
- `5 停止循环`：请求停止循环。

顶部快捷键：

- `0`：识别当前屏幕；
- `1`：请求 DeepSeek；
- `3`：点击当前答案；
- `4`：开始答题；
- `5`：停止循环。

当光标位于题目文本框、答案文本框或坐标输入框内时，数字键会正常输入，不触发快捷键。

## DeepSeek 查询逻辑

当前版本没有本地题库，查询流程固定为：

```text
题目/OCR 文本
↓
发送给 DeepSeek
↓
解析 JSON 结果
↓
显示答案和解释
↓
可选自动点击
```

DeepSeek 返回格式由程序提示词约束为：

```json
{
  "question_type": "single 或 multi 或 judge",
  "answer": "A/B/C/D/E/F、多个字母组合，或 T/F",
  "confidence": 0.0,
  "explanation": "简短解释"
}
```

程序会对返回的答案做一定清洗，例如识别 `A、C`、`ACD`、`正确/错误` 等格式，但仍不保证所有模型输出都能被正确解析。

## 练习模式自动点击

自动点击默认关闭。只有勾选“启用自动点击”后，程序才会尝试点击目标窗口。

建议配置流程：

1. 打开你自己控制的练习窗口；
2. “目标窗口”默认是 `默认前置窗口`，也可以填写稳定的窗口标题关键词；
3. 根据需要点击“框选题目区域”，只框选题干或题目主体区域；
4. 如果框选区域包含答案选项，勾选“答案/选项在题目框选内”；否则不要勾选；
5. 勾选“启用自动点击”；
6. 如需查询后立即点击，勾选“查询后自动点击”；
7. 多选题页面需要保存/提交时，勾选“多选题点击保存”；
8. 如需自动进入下一题，勾选“自动点击下一题”；
9. 建议保持“点击后鼠标复位”和“点击前双重校验”开启；
10. 点击“保存点击配置”。配置会写入 `click_config.json`。

## 答案定位逻辑

点击答案时，程序会依次尝试：

1. 根据 OCR 中的 `A/B/C/D/E/F` 或判断题“正确/错误”定位；
2. 如果选项字母定位失败，使用对应选项正文文本进行模糊匹配；
3. 如果 OCR 把字母和选项正文拆成多个文本块，程序会尝试同行合并后再匹配；
4. 如果仍然失败，重新 OCR 并重新请求 DeepSeek；
5. 如果再次失败，进入随机兜底流程；
6. 如果仍没有任何可点击目标，程序尽量不中断循环，而是记录状态并继续后续流程。

## 点击前双重坐标校验

默认启用“点击前双重校验”。同一个目标在点击前会计算两次坐标：

```text
第一次坐标 p1
↓
短暂等待
↓
第二次坐标 p2
↓
如果两次距离不大：取 p1 和 p2 的中点
↓
如果两次距离过大：第三次计算 p3，并取三次中最接近的两次的中点
```

默认阈值：

```json
"coordinate_check_max_delta_pixels": 35,
"coordinate_check_retry_delay_seconds": 0.08
```

该功能可以降低 OCR 坐标抖动、页面轻微刷新或鼠标遮挡导致的误点风险，但不能完全避免误点。

## 鼠标复位逻辑

勾选“点击后鼠标复位”后，程序在点击答案、保存或下一题后会移动鼠标，避免鼠标停在题干或选项上遮挡下一轮 OCR。

默认配置：

```json
"restore_mouse_after_click": true,
"mouse_restore_position": [0, 0]
```

当 `mouse_restore_position` 为 `[0, 0]` 时，程序会自动选择目标窗口右下角附近作为复位位置。你也可以在 `click_config.json` 中手动设置固定坐标。

## 多选题逻辑

多选题支持以下答案格式：

```text
AB
ACD
A、C、D
A C
```

多选题流程：

```text
点击识别出的多个答案
↓
如果只识别出一个答案，可随机补选一个未选项
↓
点击保存/提交/确定
↓
等待页面稳定
↓
判断是否已经自动进入下一题
↓
必要时再点击下一题
```

相关默认配置：

```json
"click_submit_after_answer": true,
"submit_button_texts": ["保存", "提交", "确定", "确认", "完成"],
"post_submit_page_wait_seconds": 1.5,
"click_next_after_submit": true,
"multi_random_extra_when_single": true,
"multi_min_selected_answers": 2
```

注意：多选题“随机补选”只是为了防止某些页面要求至少选择两个选项导致流程卡住，并不代表补选项一定正确。

## 保存按钮与下一题按钮的时序

多选题点击保存后，程序不会立刻盲目点击下一题，而是：

1. 等待 `post_submit_page_wait_seconds`；
2. 重新 OCR 判断题目是否已经变化；
3. 如果题目已变化，认为保存后已经自动进入下一题，不再额外点击“下一题”；
4. 如果题目未变化，再根据配置决定是否点击下一题；
5. 点击后再等待 `post_click_page_wait_seconds`。

这样可以降低“保存按钮已经跳题，但程序又点击下一题导致跳过一题”的风险。

## 随机兜底逻辑

当答案定位、文本匹配、重新 OCR 和重新请求 DeepSeek 后仍无法稳定点击时，程序会尝试随机兜底。

默认配置：

```json
"random_fallback_enabled": true,
"random_fallback_options": ["A", "B", "C", "D"],
"random_fallback_use_option_bounds": true
```

随机兜底优先使用 A/D 坐标作为上下边界估算选项行位置；如果边界不可用，则在 OCR 可识别的 A/B/C/D 选项中随机选择；如果仍不可用，则尝试使用手动记录的固定坐标。

随机兜底可能点击错误答案，仅用于防止练习流程中断。正式或高风险场景请关闭该功能。

## 循环模式

循环模式流程：

```text
准备目标窗口
↓
OCR 识别题目
↓
请求 DeepSeek
↓
点击答案
↓
多选题必要时点击保存
↓
必要时点击下一题
↓
鼠标复位
↓
等待下一轮
```

建议使用方式：

1. 完成目标窗口设置；
2. 框选题目区域，尽量减少 OCR 面积；
3. 设置“循环间隔秒”和“最多轮数”；
4. 点击“开始答题”；
5. 如果使用 `默认前置窗口`，程序会倒计时，请在倒计时内切到练习窗口；
6. 如需停止，点击“停止循环”。

相关默认配置：

```json
"loop_interval_seconds": 1.2,
"loop_max_rounds": 50,
"stop_on_repeated_question": true,
"same_question_grace_rounds": 2,
"post_click_page_wait_seconds": 0.6
```

`same_question_grace_rounds` 用于避免页面刷新慢时，程序因为短时间内 OCR 到同一题而过早停止。

## OCR 速度优化

建议：

1. 尽量只框选题干和必要题目文字，避免把导航栏、广告、说明文字也框进去；
2. 让目标练习窗口不要过大，减少 OCR 面积；
3. 使用目标窗口标题关键词模式，比默认前置窗口更稳定；
4. 如果页面刷新慢，适当调大“循环间隔秒”和 `post_submit_page_wait_seconds`；
5. 如果 OCR 经常误识别 A/B/C/D，可尝试把选项也包含进框选区域，并勾选“答案/选项在题目框选内”。

程序默认会在 PaddleOCR 3.x 下关闭部分文档方向分类、文档矫正和文字方向分类，并优先尝试合适的 OCR 版本，以减少初始化和识别耗时。

## 配置文件 click_config.json

常用配置项示例：

```json
{
  "enabled": false,
  "auto_click_after_query": false,
  "window_keyword": "默认前置窗口",
  "click_submit_after_answer": true,
  "click_next_after_answer": true,
  "click_next_after_submit": true,
  "restore_mouse_after_click": true,
  "double_check_click_position": true,
  "random_fallback_enabled": true,
  "multi_random_extra_when_single": true,
  "loop_interval_seconds": 1.2,
  "loop_max_rounds": 50
}
```

如果出现误点，建议优先关闭：

```json
"random_fallback_enabled": false,
"multi_random_extra_when_single": false,
"click_next_after_answer": false
```

## 常见问题

### 1. 为什么没有“导入题库 CSV”和“保存当前答案”？

当前版本已经改为 DeepSeek-only，不再使用本地题库，因此移除了题库导入、题库保存和内置题库相关功能。

### 2. 为什么 DeepSeek 已返回答案，但没有点击？

可能原因：

- 未勾选“启用自动点击”；
- 目标窗口不是当前前置窗口；
- OCR 没有识别到对应选项；
- 页面选项无法通过点击文字触发；
- 坐标双重校验结果不稳定；
- 程序判断当前前台仍是助手窗口，因此阻止点击。

### 3. 为什么多选题会多点一个随机选项？

这是当前版本的兜底策略：当 DeepSeek 或答案解析只得到一个多选答案时，为避免页面要求多选题至少选择两个答案而卡住，程序会随机补选一个未选项。该补选项可能错误，可以在配置中关闭：

```json
"multi_random_extra_when_single": false
```

### 4. 为什么程序会随机点击？

这是为了保证练习循环尽量不中断。随机兜底可能错误，建议只在自测或可容忍错误的练习环境中使用。可以在配置中关闭：

```json
"random_fallback_enabled": false
```

## 后续可扩展

如果是在你自己开发或完全控制的练习系统中使用，后续可以进一步扩展：

- 使用网页 DOM 或接口直接定位选项，而不是依赖 OCR；
- 增加题目去重和答题日志；
- 增加更细粒度的点击策略配置；
- 增加人工确认模式，DeepSeek 返回后先确认再点击。
