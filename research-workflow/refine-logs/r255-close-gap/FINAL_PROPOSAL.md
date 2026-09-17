# Round 2 Refinement: Local-Corridor Development-Conditioned Bridge Suppression

## Problem Anchor

- **Bottom-line problem**: 鍎跨鎵嬮儴 X 鍏夐楠哄垎鍓蹭腑锛屼富瑕佺矘杩炲彂鐢熷湪鑵曢/鎺岄杩戠绨囥€備綆楠ㄩ緞鍎跨楠ㄥ寲涓績灏戙€侀闂磋窛澶э紝閫氬父涓嶉渶瑕佹墿澶у墠鏅垨棰濆骞查锛涢珮楠ㄩ緞鍎跨楠ㄥ寲涓績閫愭笎鍙戣偛銆佸昂瀵稿澶с€佸眬閮ㄩ缂濈缉绐勶紝灏ゅ叾鏄杈圭晫灞€閮ㄩ€艰繎鏃跺鏄撹鍒嗗壊妯″瀷杩炴垚鍚屼竴鍓嶆櫙缁勪欢銆?- **Must-solve bottleneck**: 鍦ㄤ笉鍒囨帀鐪熷疄楠ㄥご銆佷笉鍑┖鐢熸垚楠ㄧ紳鏍囩銆佷笉璁╀綆楠ㄩ緞鐥呬緥鍙戠敓閿欒纰庤鎴栧皬楠ㄥ寲涓績娑堝け鐨勫墠鎻愪笅锛岄檷浣庨珮楠ㄩ緞杩戣窛绂婚瀵规渶绐勭摱棰堝鐨勫墠鏅疆淇★紝闃绘柇鍋囪繛鎺ャ€?- **Non-goals**: 涓嶅仛鍏ㄦ墜 component-count 浼樺寲锛涗笉浣跨敤鎺ㄧ悊鏈熼榫勶紱涓嶉€氱敤鏀剁缉鍓嶆櫙锛涗笉鐢熸垚 seam mask锛涗笉澧炲姞鎺ㄧ悊妯″潡銆?- **Constraints**: TSRS_RSNA-Epiphysis only锛涘師鏁版嵁涓嶆敼锛沘nchor 蹇呴』 y=0锛涙棤鍙潬鑳屾櫙璺宠繃锛沜lean-test-v2 閿佸畾锛沶nU-Net native base銆?- **Success condition**: 1鈥? px close-gap FP 鍜?false merge 涓嬮檷锛汸recision/Boundary 涓嶉檷锛汻ecall/Dice銆佸皬涓績 Recall 绋冲畾锛涗綆楠ㄩ緞 false split 涓嶅鍔犮€?
## Anchor and Simplicity Check

鍞竴璐＄尞淇濇寔涓猴細**璁粌鏈熷彂鑲叉潯浠跺寲鐨勫眬閮ㄨ繛缁?bridge-risk 璐熸搴﹀垎閰?*銆傝法鐥呬緥楠ㄧ紪鍙枫€乸resence head銆乺ank loss銆乧ore BCE銆佹帹鐞嗘湡 metadata銆乭ard connectivity threshold 鍧囧凡鍒犻櫎銆傞鐗堟病鏈夋柊缃戠粶鍙傛暟锛屾爣鍑?nnU-Net 鎺ㄧ悊涓嶅彉銆?
## Closed Method Definition

### 1. Sidecar source and abstention

鍘熸暟鎹殑 PNG 鏍囩瀹為檯鍖呭惈鍥惧唴瀹炰緥鍊硷紙0 涓鸿儗鏅紝1..N 涓轰笉鍚屾爣娉ㄥ疄渚嬶級锛涜繖浜涙暟鍊煎彧浣滀负鍗曞浘鍐?identity锛屼笉鍋囪璺ㄧ梾渚嬬紪鍙疯涔変竴鑷淬€?
sidecar audit 蹇呴』鎷掔粷锛?
- binary connected component 涓?instance value 鏄庢樉涓嶄竴鑷达紱
- 鍗曞疄渚嬫柇瑁備负澶氫釜澶х粍浠讹紱
- 涓ゅ疄渚嬪湪 binary label 涓凡缁忕矘杩炰笖娌℃湁 y=0锛?- 鏋佸皬浼粍浠讹紱
- 澧炲己鍚?instance split/merge锛?- pair corridor 绌胯繃绗笁瀹炰緥銆?
### 2. Separate context and distance

瀹氫箟褰掍竴鍖栧疄闄呴棿璺濓細

```text
d_ij = minimum boundary distance / min(local bone width_i, width_j)
```

涓婁笅鏂囨弿杩扮涓嶅寘鍚?d锛?
```text
u_ij = [
  orientation-normalized pair midpoint,
  relative boundary direction,
  log area ratio,
  compactness pair,
  normalized local ROI coordinate
]
```

鍙戣偛绯绘暟涓嶅啀澹扮О鈥滃紓甯告帴杩戔€濓細

```text
q_context = train-density confidence(a, sex, u_ij)
q_close   = exp(-d_ij / tau_close)
r_dev     = q_ESS(a,sex,u_ij) 脳 q_context 脳 q_close
```

tau_close 浠ョ浉瀵归瀹藉畾涔夊苟鍦?train-only Gate A 棰勬敞鍐屻€傞榫?鎬у埆鍙彁渚?training-only context density 涓?ESS锛涘疄闄呭嚑浣曡窛绂诲喅瀹氭帴杩戦闄┿€傛瘡鐥呬緥 pair 鍏堝綊涓€鍖栵紝闃叉楂橀榫?pair 鏁版洿澶氥€?
### 3. Fixed local capsule

浠?x_i*銆亁_j* 涓轰袱瀹炰緥鏈€杩戣竟鐣岀偣锛寃_i銆亀_j 涓哄眬閮ㄧ煭杞村搴︼細

```text
Omega_ij =
  Dilate(line_segment(x_i*, x_j*),
         radius = rho 脳 min(w_i,w_j))
```

棰勬敞鍐岀害鏉燂細

- capsule length <= L_max 脳 min(w_i,w_j)锛?- rho銆丩_max 鍙敤 train audit 纭畾锛?- Omega 涓庣涓夊疄渚?support 鐩镐氦鍒欐嫆缁?pair锛?- source/target eroded cores 浠庤矾寰勫唴閮ㄦ帓闄わ紱
- legal anchors = Omega 鈭?{y=0}锛?- 涓嶅厑璁?capsule 澶栫粫琛屻€?
棣栫増鍙厑璁哥洿绾?capsule锛屼笉浣跨敤鏇茬嚎璺緞銆?
### 4. Local widest-path bridge risk

鍦?Omega_ij 鐨?8-neighbor graph 鍐咃紝浠ヤ袱涓竟鐣岄偦鍩熶负 source/target锛?
```text
b_ij =
  max over P subset Omega_ij,
           |P| <= L_ij
  min over x in P excluding source/target cores
  detach(p_fg(x))
```

璺緞涓嶅緱杩涘叆绗笁瀹炰緥鎴栫寮€ capsule銆備娇鐢?maximin Dijkstra/priority flood 璁＄畻锛涜矾寰勯暱搴﹁秴杩囬娉ㄥ唽涓婇檺鍒?pair 鏃犳晥銆俠_ij 鍙綔 stop-gradient 杩炵画绯绘暟銆?
### 5. X-ray and annotation reliability

灞€閮ㄥ己搴︽柟鍚戠敱楠ㄥ唴涓?gap 鐩稿鍊肩‘瀹氾細

```text
q_valley = clip(
  [min(I_bone_i,I_bone_j)-I_gap]
  / [abs(I_bone_i-I_bone_j)+sigma_local+eps],
  0,1)

q_xray = q_valley 脳 q_double_edge
```

q_double_edge 涓轰袱渚ф柟鍚戠浉鍙嶄笖鍒嗗埆鎸囧悜涓や釜鏀寔鐨勫綊涓€鍖栨搴﹁瘉鎹€?
瀹屾暣鍍忕礌鍙潬搴︼細

```text
q_x = q_corridor 脳 q_xray 脳 q_aug 脳 q_ann
```

- q_corridor锛氳窛 capsule 涓酱鐨勮繛缁“鍑忥紱
- q_xray锛氳繛缁?[0,1]锛?- q_aug锛氫袱娆¤交寰寮哄悗 anchor 瀛樻椿涓庝綅缃竴鑷存€э紱
- q_ann锛氬疄渚?浜屽€间竴鑷淬€侀潪鏂銆侀潪绗笁楠ㄧ┛瓒婄殑杩炵画/纭粍鍚堛€?
1 px gap 鍙湁 q_xray 鍜?q_aug 鍧囬€氳繃 train-only hard floor 鎵嶄繚鐣欍€俵oss 鍙湪 full-resolution logits 璁＄畻銆?
pair 绯绘暟锛?
```text
k_ij = stopgrad(r_dev 脳 b_ij)
a_x  = max_{ij:x in pair} k_ij q_x
```

### 6. Gradient-clipped background BCE

浠?z = logit_fg - logit_bg锛実_max 涓烘渶澶у崟鍍忕礌姊害锛寊0=logit(g_max)锛?
```text
ell_gc(z) =
  softplus(z),                              if sigmoid(z) <= g_max
  softplus(z0) + g_max 脳 (z-z0),           otherwise
```

鍥犳涓ラ噸楂樼疆淇″亣妗ヤ粛鏈夊浐瀹氶潪闆剁籂姝ｆ搴?g_max锛屼絾涓嶄細鑾峰緱鏃犻檺/杩囧己鏉冮噸銆?
鎸夊浘鍍忓綊涓€鍖栵細

```text
L_close_image =
  sum_x a_x ell_gc(z_x) / (eps + sum_x a_x)

L_close = mean_valid_images(L_close_image)
L = L_nnunet_native + alpha_fixed L_close
```

鏃犳湁鏁?pair 鐨勫浘鍍忓彧浣跨敤 native loss銆傚垹闄や綆姒傜巼鍋滄闃堝€硷紝閬垮厤閲嶆柊寮曞叆闃堝€兼姈鍔ㄣ€?
### 7. One-time alpha calibration

alpha 涓嶅仛 batch-wise 鍔ㄦ€佹洿鏂般€傜敤鍥哄畾鐨?train-only calibration manifest锛?2 寮狅紝鎸夐榫勫垎灞備絾涓嶇湅 val锛夋祴閲?full-resolution logits 涓婄殑姊害鑼冩暟锛?
```text
alpha_fixed =
  clip(0.05 脳 median(g_native / [g_close+eps]),
       alpha_min, alpha_max)
```

涔嬪悗鍐荤粨銆傜洰鏍囨槸 close-gap 姊害涓綅鏁颁笉瓒呰繃 native 鐨?5%銆傝缁冧腑姊害姣斿彧浣滅洃鎺?鍋滄鏉′欢锛屼笉鍙嶉璋冭妭 alpha銆?
### 8. Exact protocol

- **Gate A**: train-only non-GPU anchor/corridor audit锛涗笉閫氳繃涓嶈缁冦€?- **Warm-up**: 2 epoch native nnU-Net锛屼繚瀛樺叡鍚?checkpoint銆?- **Gate B**: 浠庡悓涓€ warm-up checkpoint 鍒嗗弶锛?  - native continuation 2 active epochs锛?  - native + close-gap 2 active epochs銆?- **Gate C**: Gate B 閫氳繃鍚庢墠鍋?10鈥?5 epoch original-val pilot銆?- clean-test-v2 濮嬬粓閿佸畾銆?
Gate A 纭棬妲涳細

- sidecar recoverability锛?- third-instance crossing = 0锛?- widest-path capsule escape = 0锛?- 1 px/2 px anchor augmentation survival锛?- X-ray double-edge pass rate锛?- 鎸夐榫勭殑 valid pair銆丒SS銆乤ctivation锛?- 鍘熷 px銆侀澶勭悊 px銆乬ap/local-width 涓夌灏哄害銆?
Gate B/C guardrails锛?
- close-gap p_fg 鎸?1鈥?銆?鈥?銆?鈥?銆?8 px锛?- false merge锛?- Precision銆丅oundary IoU/F1锛?- per-instance Recall銆乻mall-center Recall锛?- false split銆乫ragment increase銆乫oreground area loss锛?- Dice/Recall non-inferiority銆?
## Claim Scope

棣栬疆鍙瘉鏄?local continuous bridge-risk anchor 鑳藉惁瀹夊叏闄嶄綆楂橀榫勮繎缂濆亣杩炴帴銆傚彂鑲叉潯浠跺寲鐨勮础鐚繀椤荤敱浠ヤ笅涓夐」瀵圭収鍚庢墠鑳戒富寮狅細

1. native锛?2. native + unconditioned local bridge loss锛?3. native + age/sex-conditioned local bridge loss銆?
鍚庣画鍒犻櫎 q_xray 鍜?b_ij 鍒嗗埆楠岃瘉鍥惧儚璇佹嵁涓庤繛缁ˉ椋庨櫓銆傛寮忎富寮犻渶瑕佸 seed 鎴?nnU-Net folds銆傛湭閫氳繃 Gate C 涓嶆墿灞曢骞层€佷笉杩愯 clean-test-v2銆?

