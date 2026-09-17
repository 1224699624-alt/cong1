param(
  [string]$OutputPath = "G:\gutou\YOLO+SAM\research-workflow\refine-logs\Topology_Preserving_Borrowing_YOLO_SAM_Report.pptx"
)

$ErrorActionPreference = "Stop"

function Rgb($r, $g, $b) {
  return [int]($r + ($g * 256) + ($b * 65536))
}

function Add-Text($slide, [double]$left, [double]$top, [double]$width, [double]$height, [string]$text, [int]$size, [bool]$bold=$false, [int]$color=0, [string]$font="Microsoft YaHei UI") {
  $box = $slide.Shapes.AddTextbox(1, $left, $top, $width, $height)
  $box.TextFrame2.TextRange.Text = $text
  $box.TextFrame2.TextRange.Font.Name = $font
  $box.TextFrame2.TextRange.Font.Size = $size
  $box.TextFrame2.TextRange.Font.Bold = $(if ($bold) { -1 } else { 0 })
  $box.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $color
  $box.TextFrame2.MarginLeft = 0
  $box.TextFrame2.MarginRight = 0
  $box.TextFrame2.MarginTop = 0
  $box.TextFrame2.MarginBottom = 0
  return $box
}

function Add-Title($slide, [string]$title, [string]$eyebrow="Topology Borrowing") {
  Add-Text $slide 48 32 350 24 $eyebrow 12 $true (Rgb 95 95 95) | Out-Null
  Add-Text $slide 48 66 850 54 $title 34 $true (Rgb 10 10 10) | Out-Null
  $line = $slide.Shapes.AddShape(1, 48, 126, 1184, 1.5)
  $line.Fill.ForeColor.RGB = Rgb 190 190 190
  $line.Line.Visible = 0
}

function Add-Footer($slide, [int]$num) {
  Add-Text $slide 48 690 700 16 "YOLO+SAM Epiphysis Segmentation | Topology Borrowing" 9 $false (Rgb 110 110 110) | Out-Null
  Add-Text $slide 1190 690 40 16 ([string]$num) 9 $false (Rgb 110 110 110) | Out-Null
}

function Add-Panel($slide, [double]$left, [double]$top, [double]$width, [double]$height, [string]$title, [string[]]$lines, [int]$accent) {
  $panel = $slide.Shapes.AddShape(1, $left, $top, $width, $height)
  $panel.Fill.ForeColor.RGB = Rgb 245 245 245
  $panel.Line.ForeColor.RGB = Rgb 210 210 210
  $bar = $slide.Shapes.AddShape(1, $left, $top, 6, $height)
  $bar.Fill.ForeColor.RGB = $accent
  $bar.Line.Visible = 0
  Add-Text $slide ($left+22) ($top+18) ($width-40) 28 $title 21 $true (Rgb 20 20 20) | Out-Null
  $body = ($lines | ForEach-Object { "· $_" }) -join "`r"
  $box = Add-Text $slide ($left+22) ($top+58) ($width-44) ($height-70) $body 15 $false (Rgb 55 55 55)
  $box.TextFrame2.TextRange.ParagraphFormat.SpaceAfter = 6
  return $panel
}

function Add-Step($slide, [double]$left, [double]$top, [double]$width, [string]$label, [string]$title, [string]$body, [int]$fill) {
  $shape = $slide.Shapes.AddShape(5, $left, $top, $width, 116)
  $shape.Fill.ForeColor.RGB = $fill
  $shape.Line.ForeColor.RGB = Rgb 210 210 210
  Add-Text $slide ($left+18) ($top+14) 70 18 $label 11 $true (Rgb 90 90 90) | Out-Null
  Add-Text $slide ($left+18) ($top+36) ($width-36) 26 $title 18 $true (Rgb 15 15 15) | Out-Null
  Add-Text $slide ($left+18) ($top+68) ($width-36) 38 $body 12 $false (Rgb 60 60 60) | Out-Null
}

function Add-PictureContain($slide, [string]$path, [double]$left, [double]$top, [double]$width, [double]$height, [bool]$border=$true) {
  if (!(Test-Path $path)) { return $null }
  Add-Type -AssemblyName System.Drawing
  $img = [System.Drawing.Image]::FromFile((Resolve-Path $path))
  $iw = [double]$img.Width
  $ih = [double]$img.Height
  $img.Dispose()
  $scale = [Math]::Min($width / $iw, $height / $ih)
  $w = $iw * $scale
  $h = $ih * $scale
  $x = $left + (($width - $w) / 2)
  $y = $top + (($height - $h) / 2)
  $pic = $slide.Shapes.AddPicture((Resolve-Path $path), 0, -1, $x, $y, $w, $h)
  if ($border) {
    $pic.Line.Visible = -1
    $pic.Line.ForeColor.RGB = Rgb 210 210 210
    $pic.Line.Weight = 1
  }
  return $pic
}

$pp = New-Object -ComObject PowerPoint.Application
$pp.Visible = -1
$pres = $pp.Presentations.Add()
$pres.PageSetup.SlideWidth = 1280
$pres.PageSetup.SlideHeight = 720

$blank = 12
$black = Rgb 0 0 0
$muted = Rgb 80 80 80
$orange = Rgb 255 107 53
$blue = Rgb 38 99 235
$green = Rgb 22 163 74
$red = Rgb 220 38 38

$assetRoot = Join-Path (Get-Location) "research-workflow\refine-logs\topology_paper_assets"
$roadOverview = Join-Path $assetRoot "01_road_segmentation_satloss_cell008_out01_001.png"
$roadResults = Join-Path $assetRoot "01_road_segmentation_satloss_cell048_out00_033.png"
$crackExample = Join-Path $assetRoot "02_crack_segmentation_cldice_satloss_cell031_out00_009.png"
$celegansExample = Join-Path $assetRoot "03_celegans_transfer_learning_cell004_out01_001.png"
$driveExample = Join-Path $assetRoot "04_drive_retinal_vessel_transfer_learning_cell003_out01_001.png"

# 1 Cover
$s = $pres.Slides.Add(1, $blank)
$s.Background.Fill.ForeColor.RGB = Rgb 255 255 255
Add-Text $s 52 46 360 24 "YOLO+SAM 骨骺分割项目汇报" 13 $true (Rgb 95 95 95) | Out-Null
Add-Text $s 52 150 720 150 "从拓扑保持到骨缝分离" 62 $true $black | Out-Null
Add-Text $s 56 306 650 88 "借鉴 Topology-Preserving Image Segmentation with Spatial-Aware Persistent Feature Matching 的结构一致性思想" 24 $false $muted | Out-Null
$accent = $s.Shapes.AddShape(1, 52, 450, 480, 8)
$accent.Fill.ForeColor.RGB = $orange
$accent.Line.Visible = 0
Add-Text $s 56 490 690 72 "目标：保持 Dice/IoU 竞争力，同时减少骨缝粘连、误连接和边界错误。" 21 $true (Rgb 25 25 25) | Out-Null
$heroFrame = $s.Shapes.AddShape(1, 810, 115, 365, 410)
$heroFrame.Fill.ForeColor.RGB = Rgb 248 248 248
$heroFrame.Line.ForeColor.RGB = Rgb 220 220 220
Add-PictureContain $s $roadOverview 835 148 315 190 $false | Out-Null
Add-PictureContain $s $driveExample 835 360 315 95 $false | Out-Null
Add-Text $s 835 476 315 34 "论文示例：道路/血管等细长结构强调拓扑连续性" 12 $false (Rgb 75 75 75) | Out-Null
Add-Footer $s 1

# 2 Communication job
$s = $pres.Slides.Add(2, $blank)
Add-Title $s "这篇工作的价值在于把结构错误放到训练目标里"
Add-Panel $s 60 155 340 245 "传统分割的盲点" @("Dice 和 IoU 可以很高", "细长结构仍可能断裂或误连", "结构错误会影响下游判断") $red | Out-Null
Add-Panel $s 470 155 340 245 "论文提供的启发" @("用 persistent homology 描述连通与孔洞", "用空间感知匹配约束拓扑特征", "把拓扑一致性作为显式监督") $blue | Out-Null
Add-Panel $s 880 155 340 245 "迁移到骨骺任务" @("我们关心的不是血管/道路连通", "而是骨缝背景是否连通", "拓扑约束方向需要反过来") $green | Out-Null
$strip = $s.Shapes.AddShape(1, 74, 430, 1120, 110)
$strip.Fill.ForeColor.RGB = Rgb 250 250 250
$strip.Line.ForeColor.RGB = Rgb 225 225 225
Add-PictureContain $s $roadOverview 92 445 250 72 $false | Out-Null
Add-PictureContain $s $crackExample 365 445 250 72 $false | Out-Null
Add-PictureContain $s $celegansExample 638 445 250 72 $false | Out-Null
Add-PictureContain $s $driveExample 911 445 250 72 $false | Out-Null
Add-Text $s 92 520 250 16 "Road topology" 10 $true (Rgb 85 85 85) | Out-Null
Add-Text $s 365 520 250 16 "Crack connectivity" 10 $true (Rgb 85 85 85) | Out-Null
Add-Text $s 638 520 250 16 "C. elegans transfer" 10 $true (Rgb 85 85 85) | Out-Null
Add-Text $s 911 520 250 16 "Retinal vessels" 10 $true (Rgb 85 85 85) | Out-Null
$summaryText = "核心结论：论文的保持结构不是照搬 loss，而是启发我们定义更贴近骨缝失败模式的拓扑目标。"
Add-Text $s 74 568 1100 54 $summaryText 22 $true $black | Out-Null
Add-Footer $s 2

# 3 Paper core
$s = $pres.Slides.Add(3, $blank)
Add-Title $s "论文方法围绕拓扑特征匹配，而不是只优化像素重叠"
Add-Text $s 62 160 480 44 "方法组成" 26 $true $black | Out-Null
Add-Step $s 62 220 245 "1" "像素监督" "BCE / Dice 提供区域重叠与分类基础" (Rgb 245 245 245)
Add-Step $s 337 220 245 "2" "骨架连通" "clDice 关注中心线与细长结构连续性" (Rgb 245 245 245)
Add-Step $s 612 220 245 "3" "拓扑监督" "Persistent homology 提取连通分量和孔洞" (Rgb 245 245 245)
Add-Step $s 887 220 245 "4" "空间匹配" "SATLoss 用位置约束匹配拓扑特征" (Rgb 245 245 245)
$paperCoreText = "可借鉴的不是某个单独公式，而是评价方式的转变：从像素是否重叠，扩展到预测结构是否符合任务拓扑。"
$formula = $s.Shapes.AddShape(1, 94, 382, 470, 92)
$formula.Fill.ForeColor.RGB = Rgb 255 247 237
$formula.Line.ForeColor.RGB = $orange
Add-Text $s 120 405 420 26 "L = BCE + Dice + clDice + SATLoss" 23 $true $black | Out-Null
Add-Text $s 120 438 410 20 "像素、区域、骨架、拓扑同时进入训练目标" 14 $false (Rgb 80 80 80) | Out-Null
Add-PictureContain $s $roadResults 650 372 500 155 $true | Out-Null
Add-Text $s 650 535 500 20 "论文实验图：不同损失下的结构连续性对比" 12 $false (Rgb 90 90 90) | Out-Null
Add-Text $s 86 575 1080 54 $paperCoreText 22 $true $black | Out-Null
Add-Footer $s 3

# 4 Direct copy mismatch
$s = $pres.Slides.Add(4, $blank)
Add-Title $s "直接鼓励前景连通，可能会加重骨缝粘连"
Add-Panel $s 70 155 500 250 "论文原始场景" @("道路、裂纹、血管是连续细长前景", "断裂是主要错误", "拓扑目标通常鼓励前景连通") $blue | Out-Null
Add-Panel $s 710 155 500 250 "骨骺分割场景" @("相邻骨之间应由背景骨缝隔开", "误连是主要错误之一", "盲目保持前景连通会把粘连固化") $orange | Out-Null
Add-PictureContain $s $crackExample 104 430 200 78 $true | Out-Null
Add-PictureContain $s $driveExample 328 430 200 78 $true | Out-Null
$boneA = $s.Shapes.AddShape(9, 790, 435, 120, 92)
$boneA.Fill.ForeColor.RGB = Rgb 35 35 35
$boneA.Line.Visible = 0
$boneB = $s.Shapes.AddShape(9, 990, 435, 120, 92)
$boneB.Fill.ForeColor.RGB = Rgb 35 35 35
$boneB.Line.Visible = 0
$badBridge = $s.Shapes.AddShape(1, 900, 468, 105, 22)
$badBridge.Fill.ForeColor.RGB = Rgb 35 35 35
$badBridge.Line.Visible = 0
Add-Text $s 785 530 330 22 "错误借鉴会把相邻骨粘成一个前景连通块" 14 $true $red | Out-Null
$reverseText = "所以我们的借鉴方向是反向拓扑：不奖励骨头连在一起，而奖励骨缝背景通道保持连通。"
Add-Text $s 96 560 1060 46 $reverseText 25 $true $black | Out-Null
Add-Footer $s 4

# 5 Reversed topology diagram
$s = $pres.Slides.Add(5, $blank)
Add-Title $s "我们把拓扑目标从前景连通翻转为背景通道连通"
$leftPanel = $s.Shapes.AddShape(1, 80, 180, 470, 330)
$leftPanel.Fill.ForeColor.RGB = Rgb 248 248 248
$leftPanel.Line.ForeColor.RGB = Rgb 210 210 210
$rightPanel = $s.Shapes.AddShape(1, 730, 180, 470, 330)
$rightPanel.Fill.ForeColor.RGB = Rgb 248 248 248
$rightPanel.Line.ForeColor.RGB = Rgb 210 210 210
Add-Text $s 108 205 400 30 "前景拓扑保持" 24 $true $black | Out-Null
Add-Text $s 758 205 400 30 "骨缝背景连通" 24 $true $black | Out-Null
foreach($x in @(170, 315)) {
  $o = $s.Shapes.AddShape(9, $x, 292, 120, 120)
  $o.Fill.ForeColor.RGB = Rgb 40 40 40
  $o.Line.Visible = 0
}
$bridge = $s.Shapes.AddShape(1, 265, 345, 100, 28)
$bridge.Fill.ForeColor.RGB = Rgb 40 40 40
$bridge.Line.Visible = 0
Add-Text $s 118 445 380 36 "适合血管、道路、裂纹：断裂需要被修复" 17 $false $muted | Out-Null
foreach($x in @(820, 990)) {
  $o = $s.Shapes.AddShape(9, $x, 292, 120, 120)
  $o.Fill.ForeColor.RGB = Rgb 40 40 40
  $o.Line.Visible = 0
}
$gap = $s.Shapes.AddShape(1, 944, 270, 22, 170)
$gap.Fill.ForeColor.RGB = Rgb 255 255 255
$gap.Line.ForeColor.RGB = $orange
$gap.Line.Weight = 3
Add-Text $s 768 445 390 36 "适合骨骺：骨缝应作为背景通道把相邻骨分开" 17 $false $muted | Out-Null
$arrow = $s.Shapes.AddConnector(1, 585, 345, 700, 345)
$arrow.Line.ForeColor.RGB = $orange
$arrow.Line.Weight = 3
$arrow.Line.EndArrowheadStyle = 3
Add-Text $s 580 305 130 28 "反向借鉴" 18 $true $orange | Out-Null
Add-Footer $s 5

# 6 Metric mapping
$s = $pres.Slides.Add(6, $blank)
Add-Title $s "评价指标也要从重叠指标扩展到失败模式诊断"
Add-Panel $s 60 165 350 380 "主表保持竞争力" @("Dice", "IoU / Jaccard", "Precision / Recall", "目标：不明显低于 ARAA/R110") $blue | Out-Null
Add-Panel $s 465 165 350 380 "边界与表面质量" @("Boundary IoU", "Boundary F1", "Surface Dice 2px / 5px", "HD95 / ASSD") $green | Out-Null
Add-Panel $s 870 165 350 380 "骨缝与解剖一致性" @("gap-region FP rate", "component merge rate", "component count MAE", "定性检查过度腐蚀和漏分") $orange | Out-Null
Add-Text $s 74 580 1080 38 "这套指标让拓扑思想落到骨骺任务：不是证明预测更黑或更细，而是证明骨缝分离更合理且骨结构仍完整。" 21 $true $black | Out-Null
Add-Footer $s 6

# 7 Experiment route
$s = $pres.Slides.Add(7, $blank)
Add-Title $s "当前实验路线是在寻找可部署的背景连通选择器"
Add-Step $s 60 170 210 "R240" "几何门控" "边界/gap 有小幅正向，但 foreground cut 太多" (Rgb 250 250 250)
Add-Step $s 300 170 210 "R241" "表格特征评分" "AUC/AP 不足，无法分离安全 cut 与风险 cut" (Rgb 250 250 250)
Add-Step $s 540 170 210 "R242" "局部 crop scorer" "有弱 useful 信号，但 risk head 仍不可靠" (Rgb 250 250 250)
Add-Step $s 780 170 210 "R244" "merge-targeted 生成器" "oracle 信号变强，候选更接近骨缝误连" (Rgb 250 250 250)
Add-Step $s 1020 170 190 "R245-R248" "筛选器升级" "scalar filter no-go，等待 full 后进入上下文评分" (Rgb 250 250 250)
$rule = $s.Shapes.AddShape(1, 165, 350, 930, 3)
$rule.Fill.ForeColor.RGB = $orange
$rule.Line.Visible = 0
Add-Text $s 82 405 1080 78 "这条路线的关键判断：候选生成器已经能产生安全骨缝 cut，但部署时仍难从风险 cut 中选出来。" 27 $true $black | Out-Null
Add-Footer $s 7

# 8 Evidence table
$s = $pres.Slides.Add(8, $blank)
Add-Title $s "最新证据显示路线有上界，但选择器还不够安全"
$headers = @("实验", "有利信号", "主要问题", "结论")
$rows = @(
  @("R240", "Boundary IoU +0.000107`ngap FP -0.000230", "mean cut GT foreground > 0.5", "几何门控太冒险"),
  @("R241", "尝试 GT-free 特征学习", "HGB AUC 0.468`n高分仍多为风险 cut", "表格特征不够"),
  @("R242", "8190 crops, 765 safe", "relaxed 仍 1 safe / 14 risk", "crop scorer 未过门"),
  @("R244 partial40", "oracle BIoU +0.001240`ncomponent MAE -0.7778", "GT-free clean 36 safe / 201 risk", "生成器有潜力")
)
$x = 54; $y = 155; $w = @(170, 310, 340, 310); $hHead = 42; $hRow = 92
for($i=0; $i -lt 4; $i++) {
  $cell = $s.Shapes.AddShape(1, $x, $y, $w[$i], $hHead)
  $cell.Fill.ForeColor.RGB = Rgb 30 30 30
  $cell.Line.ForeColor.RGB = Rgb 255 255 255
  Add-Text $s ($x+10) ($y+10) ($w[$i]-20) 22 $headers[$i] 15 $true (Rgb 255 255 255) | Out-Null
  $x += $w[$i]
}
for($r=0; $r -lt $rows.Count; $r++) {
  $x = 54; $y = 197 + $r*$hRow
  for($c=0; $c -lt 4; $c++) {
    $cell = $s.Shapes.AddShape(1, $x, $y, $w[$c], $hRow)
    $cell.Fill.ForeColor.RGB = $(if($r % 2 -eq 0){Rgb 248 248 248}else{Rgb 238 238 238})
    $cell.Line.ForeColor.RGB = Rgb 255 255 255
    Add-Text $s ($x+10) ($y+12) ($w[$c]-20) ($hRow-20) $rows[$r][$c] 14 $($c -eq 0) (Rgb 35 35 35) | Out-Null
    $x += $w[$c]
  }
}
Add-Footer $s 8

# 9 Failure analysis
$s = $pres.Slides.Add(9, $blank)
Add-Title $s "真正的瓶颈不是拓扑思想，而是安全选择"
Add-Panel $s 74 165 520 360 "已经成立的部分" @("oracle-safe cut 能降低 gap FP", "component count MAE 明显改善", "Recall 可以保持不下降", "说明背景连通目标方向是有价值的") $green | Out-Null
Add-Panel $s 686 165 520 360 "还没解决的部分" @("GT-free 规则会切到真实骨前景", "scalar 特征无法区分安全骨缝与浅层骨切割", "局部 crop scorer 仍被风险样本主导", "还不能写 mask 或上 clean-test-v2") $red | Out-Null
$failureText = "下一步必须提升选对 cut 的能力，而不是继续扩大切割或放宽阈值。"
Add-Text $s 92 565 1050 44 $failureText 26 $true $black | Out-Null
Add-Footer $s 9

# 10 Next experiments
$s = $pres.Slides.Add(10, $blank)
Add-Title $s "下一轮实验应围绕更强的上下文选择器展开"
Add-Panel $s 60 160 360 390 "短期门控" @("等 R244 full original-val 完成", "R249 自动跑 full R245/R246", "若 R245 no-go，R250 自动接 R248", "全程不使用 clean-test-v2 调参") $blue | Out-Null
Add-Panel $s 460 160 360 390 "模型改进方向" @("局部上下文评分器", "显式 foreground erosion 风险头", "component-aware preselection", "候选生成器先减少风险分布") $orange | Out-Null
Add-Panel $s 860 160 360 390 "通过条件" @("Dice/IoU 不明显退化", "Boundary IoU/F1 和 Surface Dice 提升", "gap FP 与 component error 下降", "可视化证明不是过度腐蚀") $green | Out-Null
Add-Footer $s 10

# 11 Close
$s = $pres.Slides.Add(11, $blank)
Add-Title $s "汇报结论：借鉴拓扑保持，但目标要为骨缝任务重写"
Add-Text $s 72 170 1030 86 "Topology-preserving 的核心启发是把结构一致性显式纳入目标；在骨骺分割中，我们应把它重写为骨缝背景连通与相邻骨分离。" 31 $true $black | Out-Null
Add-Panel $s 80 300 330 210 "当前判断" @("方向仍然活着", "R244 oracle 信号强", "部署选择器尚未过门") $orange | Out-Null
Add-Panel $s 475 300 330 210 "不能宣称" @("不能说已解决粘连", "不能用 clean-test-v2 选阈值", "不能用腐蚀换指标") $red | Out-Null
Add-Panel $s 870 300 330 210 "可以推进" @("等 full R244/R245/R246", "用 R248 或更强上下文模型筛选", "用 R201 指标和 hard cases 审计") $green | Out-Null
Add-Text $s 76 612 1070 36 "来源：Topology-Preserving Image Segmentation with Spatial-Aware Persistent Feature Matching GitHub README；项目 R240/R241/R242/R244/R245/R246/R249 日志。" 12 $false (Rgb 100 100 100) | Out-Null
Add-Footer $s 11

$outDir = Split-Path -Parent $OutputPath
if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
if (Test-Path $OutputPath) { Remove-Item $OutputPath -Force }
$pres.SaveAs($OutputPath)

# Export PNG previews next to the deck for QA.
$previewDir = Join-Path $outDir "Topology_Preserving_Borrowing_YOLO_SAM_Report_preview"
if (Test-Path $previewDir) { Remove-Item -Recurse -Force $previewDir }
New-Item -ItemType Directory -Force -Path $previewDir | Out-Null
$pres.Export($previewDir, "PNG", 1280, 720)

$pres.Close()
$pp.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp) | Out-Null
Write-Output $OutputPath
