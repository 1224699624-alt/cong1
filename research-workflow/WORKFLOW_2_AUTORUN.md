# Workflow 2锛氳嚜鍔ㄧ鐮斿惊鐜紙鐫′竴瑙夐啋鏉ョ湅缁撴灉锛?
## 鍏堣缁撹

瑕佸仛鍒颁綘璇寸殑杩欑鏁堟灉锛屾牳蹇冧笉鏄竴鍙?`/auto-review-loop` 鍛戒护锛岃€屾槸涓夊眰涓滆タ涓€璧峰埌浣嶏細

1. **瀹為獙鏈韩瑕佽劚绂?Codex 浼氳瘽鐙珛杩愯**
2. **椤圭洰鐘舵€佽钀界洏锛屼笅涓€娆?Codex 杩涙潵鑳芥帴鐫€骞?*
3. **闇€瑕佷竴涓閮ㄨЕ鍙戝櫒瀹氭湡鎶?Codex 鍐嶅彨璧锋潵**

鍙湁绗?1 灞傦紝娌℃湁鈥滅潯閱掔湅缁撴灉鈥濓紝鍥犱负瀹為獙璺戝畬鍚庢病浜烘敹灏俱€? 
鍙湁绗?2 灞傦紝娌℃湁鈥滆嚜鍔ㄥ惊鐜€濓紝鍥犱负鐘舵€佽櫧鐒跺湪锛屼絾娌′汉缁х画璇汇€? 
鍙湁绗?3 灞傦紝娌℃湁鍓嶄袱灞傦紝涔熷彧鏄弽澶嶇┖杞€?
## 杩欏拰鈥滄垜璁╀綘鑷繁涓€鐩磋窇鍒伴搴︽仮澶嶁€濅负鍟ヤ笉鏄竴鍥炰簨

褰撳墠 Codex 浼氳瘽鏈韩**涓嶄細鍦ㄩ搴﹂噸缃悗鑷姩鑻忛啋**銆? 
鎵€浠ョ湡瑕佸仛鈥滅潯閱掔湅缁撴灉鈥濓紝蹇呴』鐢ㄤ竴涓?*澶栭儴寰幆鍣?*鍙嶅璋冪敤 Codex銆?
杩欎釜浠撳簱閲屾垜宸茬粡琛ヤ簡杩欏鏈哄埗鐨勪袱涓叧閿枃浠讹細

- 鑷姩缁窇 prompt锛?  [WORKFLOW2_AUTORUN_PROMPT.md](/g:/gutou/YOLO+SAM/research-workflow/WORKFLOW2_AUTORUN_PROMPT.md)
- 鑷姩缁窇鑴氭湰锛?  [run_workflow2_loop.ps1](/g:/gutou/YOLO+SAM/scripts/run_workflow2_loop.ps1)

## 鎺ㄨ崘鍋氭硶

### 鏂规 A锛氭渶绋崇殑鈥滅潯涓€瑙夐啋鏉ョ湅缁撴灉鈥濇柟妗?
鐢?`codex exec` 闈炰氦浜掑惊鐜紝鑰屼笉鏄緷璧栦氦浜掑紡 `codex resume`銆?
鍘熷洜锛?
- `exec` 鏇撮€傚悎璁″垝浠诲姟 / 鍚庡彴瀹堟姢
- 姣忎竴杞兘浼氶€€鍑猴紝渚夸簬涓嬫鑷姩鍐嶆媺璧?- 褰撳墠浠撳簱宸茬粡鎶婄姸鎬佸啓杩涳細
  - `research-workflow/refine-logs/AUTONOMOUS_LOOP_STATUS.md`
  - `research-workflow/refine-logs/EXPERIMENT_TRACKER.md`
  - `research-workflow/refine-logs/EXPERIMENT_RESULTS.md`

涔熷氨鏄锛?*鐘舵€侀潬鏂囦欢缁紝涓嶉潬涓婁笅鏂囩獥鍙ｇ画**銆?
### 鏂规 B锛氭墜鍔ㄧ画璺戠増

濡傛灉浣犲彧鎯冲伓灏斿洖鏉ヨ嚜宸辩偣涓€涓嬶細

```powershell
codex resume 019ecb56-a5f9-7712-af06-b82cdbec5635 "缁х画褰撳墠 ARIS 鐮旂┒寰幆锛屽厛璇?AUTONOMOUS_LOOP_STATUS.md 鍐嶈鍔?
```

杩欎釜閫傚悎浜虹洴鐫€锛屼笉閫傚悎鐪熸鏃犱汉鍊煎畧銆?
## 涓€閿紑璺戞柟寮?
鍦ㄤ粨搴撴牴鐩綍鎵ц锛?
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_workflow2_loop.ps1 `
  -RepoRoot "G:\gutou\YOLO+SAM" `
  -IntervalMinutes 20 `
  -MaxPasses 999 `
  -Model "5.6-sol"
```

杩欎細鍋氱殑浜嬶細

1. 璇诲彇 `WORKFLOW2_AUTORUN_PROMPT.md`
2. 鐢?`codex exec` 鍦ㄤ粨搴撴牴鐩綍璺戜竴杞?3. 鎶婅緭鍑鸿鍒?`research-workflow/daemon-logs/`
4. 鏃犺鎴愬姛銆佸け璐ャ€侀搴︽姤閿欙紝閮戒細绛夊緟涓€娈垫椂闂村啀璺戜笅涓€杞?5. 涓嬩竴杞户缁牴鎹粨搴撻噷鐨勭姸鎬佹枃浠舵帴鐫€鎺ㄨ繘

## 鍏堝仛涓€娆℃湰鍦拌嚜妫€

鍦ㄧ湡姝ｅ悗鍙拌窇涔嬪墠锛屽缓璁厛鎵ц锛?
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test_workflow2_setup.ps1 `
  -RepoRoot "G:\gutou\YOLO+SAM"
```

瀹冧細妫€鏌ワ細

- `codex` / `codex exec` / `codex resume` 鏄惁鍙敤
- Workflow 2 鎵€闇€鐨?prompt銆佽鏄庢枃妗ｃ€佺姸鎬佹枃浠舵槸鍚﹀瓨鍦?- `daemon-logs` 鐩綍鏄惁灏辩华

## 鍏堝仛涓€娆℃棤鎹熺儫闆炬祴璇?
濡傛灉浣犳兂鍏堢‘璁ゅ惊鐜櫒鏈韩鑳借窇閫氾紝浣嗗張涓嶆兂鐪熺殑璋冪敤妯″瀷锛?
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_workflow2_loop.ps1 `
  -RepoRoot "G:\gutou\YOLO+SAM" `
  -MaxPasses 1 `
  -DryRun
```

杩欎釜妯″紡浼氾細

- 姝ｅ父鍒涘缓 daemon log
- 姝ｅ父璧颁竴杞惊鐜?- 浣嗕笉浼氱湡姝ｈ皟鐢?`codex exec` / `codex resume`

寰堥€傚悎鍏堥獙璇佷换鍔¤鍒掑拰鍚庡彴鍚姩閾捐矾銆?
## 涓€閿鎴愬紑鏈鸿嚜鍚换鍔?
濡傛灉浣犳兂鐪熸鍋氬埌鈥滄満鍣ㄥ紑鐫€灏辫嚜宸辩户缁€濓紝鍙互鐩存帴瀹夎 Windows 璁″垝浠诲姟锛?
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_workflow2_task.ps1 `
  -RepoRoot "G:\gutou\YOLO+SAM" `
  -TaskName "YOLO_SAM_Workflow2_Autorun" `
  -IntervalMinutes 20 `
  -Model "5.6-sol"
```

瀹夎鍚庡父鐢ㄥ懡浠わ細

```powershell
Start-ScheduledTask -TaskName "YOLO_SAM_Workflow2_Autorun"
Get-ScheduledTask -TaskName "YOLO_SAM_Workflow2_Autorun" | Get-ScheduledTaskInfo
Unregister-ScheduledTask -TaskName "YOLO_SAM_Workflow2_Autorun" -Confirm:$false
```

## 濡傛灉鎯宠鑷姩閫氱煡

褰撳墠鎴戝凡缁忚ˉ浜嗕竴涓€氱敤 webhook 閫氱煡鑴氭湰锛?
- [send_workflow2_notification.ps1](/g:/gutou/YOLO+SAM/scripts/send_workflow2_notification.ps1)

鏈€绠€鍗曠殑鐢ㄦ硶鏄缃幆澧冨彉閲忥細

```powershell
$env:WORKFLOW2_WEBHOOK_URL = "https://your-webhook-url"
```

鐒跺悗鍚姩寰幆鏃跺姞锛?
```powershell
-NotifyOnEveryPass
```

濡傛灉浣犱笉鎯虫瘡杞兘閫氱煡锛屽彧淇濈暀寮€濮?/ 寮傚父 / 缁撴潫閫氱煡锛屼笉鍔犺繖涓弬鏁颁篃鍙互銆?
## 濡傛灉鎯崇湡姝ｅ悗鍙拌窇

鍙互鐢?PowerShell 鍚庡彴鍚姩锛?
```powershell
Start-Process powershell `
  -WindowStyle Hidden `
  -ArgumentList '-ExecutionPolicy Bypass -File "G:\gutou\YOLO+SAM\scripts\run_workflow2_loop.ps1" -RepoRoot "G:\gutou\YOLO+SAM" -IntervalMinutes 20 -MaxPasses 999 -Model "5.6-sol"'
```

## 鏈€鍏抽敭鐨勭姸鎬佹枃浠?
鎯冲仛鍒?Workflow 2锛孋odex 姣忚疆閮藉繀椤诲厛璇昏繖浜涳細

- [RESEARCH_BRIEF.md](/g:/gutou/YOLO+SAM/research-workflow/RESEARCH_BRIEF.md)
- [AUTONOMOUS_LOOP_STATUS.md](/g:/gutou/YOLO+SAM/research-workflow/refine-logs/AUTONOMOUS_LOOP_STATUS.md)
- [EXPERIMENT_TRACKER.md](/g:/gutou/YOLO+SAM/research-workflow/refine-logs/EXPERIMENT_TRACKER.md)
- [EXPERIMENT_RESULTS.md](/g:/gutou/YOLO+SAM/research-workflow/refine-logs/EXPERIMENT_RESULTS.md)
- [NEXT_CANDIDATE_REPAIR_PLAN.md](/g:/gutou/YOLO+SAM/research-workflow/refine-logs/NEXT_CANDIDATE_REPAIR_PLAN.md)

杩欎簺鏂囦欢鐨勪綔鐢ㄥ垎鍒槸锛?
- `RESEARCH_BRIEF.md`: 鎬荤洰鏍囦笉婕傜Щ
- `AUTONOMOUS_LOOP_STATUS.md`: 褰撳墠璺戝埌鍝竴姝?- `EXPERIMENT_TRACKER.md`: 鍝釜 run 鍦ㄨ窇銆佸摢涓琛?- `EXPERIMENT_RESULTS.md`: 涓婁竴杞疄楠屽埌搴曟敮鎸佷簡浠€涔?- `NEXT_CANDIDATE_REPAIR_PLAN.md`: 涓嬩竴杞敼浠€涔堬紝涓嶇敤姣忔閲嶆兂

## 鐜板湪杩欏鏈哄埗绂烩€滅潯閱掔湅缁撴灉鈥濊繕宸粈涔?
褰撳墠浠撳簱宸茬粡鏈夛細

- 杩滅 detached 瀹為獙杩愯
- ARIS 鐘舵€佽惤鐩?- 涓嬩竴杞€欓€変慨琛ヨ鍒?- 鑷姩缁窇 prompt
- 鑷姩缁窇鑴氭湰

杩樺彲浠ョ户缁寮虹殑涓ら」锛?
1. **閫氱煡**
   鏈€濂藉啀鎺ヤ竴涓?Feishu / 閭欢 / Telegram 閫氱煡
   杩欐牱瀹為獙缁撴潫鎴?verdict 鍙樺寲鏃朵綘涓嶇敤鎵嬪姩鐪嬫棩蹇?
2. **璁″垝浠诲姟**
   鎶?`run_workflow2_loop.ps1` 鏀捐繘 Windows Task Scheduler
   杩欐牱灏辩畻褰撳墠 PowerShell 绐楀彛鍏充簡锛屾満鍣ㄥ紑鐫€瀹冧篃鑳界户缁媺璧?
## 鐪熸鍙揪鍒扮殑鏁堟灉

濡傛灉浣犳妸杩欏鏂瑰紡寮€璧锋潵锛屾晥鏋滀細鏄細

- 浣犵潯鍓嶆妸瀹為獙鎸傚埌杩滅
- 鏈満姣忛殧 20 鍒嗛挓鑷姩鍙竴娆?Codex
- Codex 璇荤姸鎬佹枃浠讹紝妫€鏌ュ疄楠岋紝鏀剁粨鏋滐紝鏇存柊缁撹
- 闇€瑕佷笅涓€杞椂缁х画鏀逛唬鐮佸苟鍙戣溅
- 浣犻啋鏉ユ椂鐪嬪埌鐨勪笉鏄€滀竴涓缁冩棩蹇椻€濓紝鑰屾槸涓€鏁村锛?  - 褰撳墠杩愯鐘舵€?  - 宸插畬鎴愮粨鏋?  - verdict
  - 涓嬩竴姝ユ槸鍚﹀凡缁忚嚜鍔ㄥ彂杞?
## 涓€鍙ヨ瘽绛旀

瑕佸仛鍒扳€淲orkflow 2锛氳嚜鍔ㄧ鐮斿惊鐜紙鐫′竴瑙夐啋鏉ョ湅缁撴灉锛夆€濓紝浣犻渶瑕佹妸 **杩滅瀹為獙杩愯銆丄RIS 鐘舵€佽惤鐩樸€佸閮ㄥ畾鏃堕噸鏂拌皟鐢?Codex** 杩欎笁浠朵簨杩炶捣鏉ワ紱杩欎釜浠撳簱鎴戝凡缁忚ˉ鍒颁簡鍓嶄袱浠跺崐锛岀幇鍦ㄦ渶鐩存帴鐨勮惤鍦版柟寮忓氨鏄窇 [run_workflow2_loop.ps1](/g:/gutou/YOLO+SAM/scripts/run_workflow2_loop.ps1)銆?
## 濡傛灉浣犺鐨勬槸鈥滆缁冭窇瀹屽悗鑷姩杩涘叆涓婃父 /auto-review-loop鈥?
杩欎欢浜嬪拰鏅€氱殑 Workflow 2 杞涓嶅畬鍏ㄤ竴鏍枫€?
姝ｇ‘鍋氭硶鏄細

1. `experiment-bridge` 璐熻矗鎶婂疄楠屾寕鍒拌繙绔苟鎸佺画鏇存柊 `EXPERIMENT_TRACKER.md`
2. 澶栭儴鐩戞帶鍣ㄥ彧绛夊緟 **bridge 澶栭儴浜嬪疄瀹屾垚**
3. 涓€鏃?bridge 瀵瑰簲 run 鍦?tracker 閲屽彉鎴?`DONE`锛屽啀瑙﹀彂涓€娆′笂娓?Workflow 2

鎴戝凡缁忚ˉ浜嗚繖涓笓闂ㄧ殑妗ユ帴鐩戞帶鍣細

- [monitor_bridge_then_autoreview.ps1](/g:/gutou/YOLO+SAM/scripts/monitor_bridge_then_autoreview.ps1)
- 瀹冧娇鐢ㄧ殑 trigger prompt锛?  [AUTO_REVIEW_TRIGGER_PROMPT.md](/g:/gutou/YOLO+SAM/research-workflow/AUTO_REVIEW_TRIGGER_PROMPT.md)

鎺ㄨ崘鐢ㄦ硶锛?
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\monitor_bridge_then_autoreview.ps1 `
  -RepoRoot "G:\gutou\YOLO+SAM" `
  -RunId "R011" `
  -Model "5.6-sol" `
  -PollSeconds 300
```

杩欎細鍋氱殑浜嬶細

- 鍙嶅妫€鏌?`EXPERIMENT_TRACKER.md`
- 鍙 `R011` 杩樹笉鏄?`DONE`锛屽氨缁х画绛?- 涓€鏃?`R011` 鍦?bridge 闃舵瀹屾垚锛屽氨鑷姩鐢?`codex exec` 瑙﹀彂涓婃父 Workflow 2
- 涔熷氨鏄嚜鍔ㄨ繘鍏?**涓婃父 skill 璇箟涓嬬殑 `/auto-review-loop`**

