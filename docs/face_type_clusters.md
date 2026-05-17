# Face Type Clustering Report

This is an exploratory clustering of idol face embeddings, not an objective taxonomy.
Known cross-group aliases are collapsed before clustering, and female/male members are clustered separately.

## Method

- Feature: normalized ArcFace vector concatenated with 0.5 * normalized FaRL vector
- Filter: member vectors with at least the requested minimum images/confidence
- Clustering: kmeans, cluster count selected near the target size by silhouette/balance
- Contact sheets: local representative crops nearest to each cluster centroid

## F Clusters

- Members: 389
- Clusters: 10
- Silhouette: 0.0239

### F01 (52 members)

- Avg images/confidence: 7.3 / 0.604
- Top groups: USPEER(6), LE SSERAFIM(2), H1-KEY(2), ILLIT(2), BABYMONSTER(2)
- Representatives: USPEER Seoyu, BABYMONSTER Ruka, BADVILLAIN Kelly, Hearts2Hearts Stella, UNCHILD Tina, USPEER Daon, iii Ahra, SAY MY NAME Junhwi
- Contact sheet: `.cache/face_type_clusters/F01.jpg`

### F02 (48 members)

- Avg images/confidence: 7.8 / 0.605
- Top groups: Kep1er(4), NiziU(3), ICHILLIN'(3), izna(2), XG(2)
- Representatives: Kep1er Chaehyun, ICHILLIN' Chaerin, ICHILLIN' Joonie, LIGHTSUM Sangah, NMIXX Kyujin, BEWAVE Yunseul, Kep1er Dayeon, ITZY Yeji
- Contact sheet: `.cache/face_type_clusters/F02.jpg`

### F03 (45 members)

- Avg images/confidence: 7.9 / 0.579
- Top groups: FIFTY FIFTY(4), Queenz Eye(3), KISS OF LIFE(2), BABYMONSTER(2), Billlie(2)
- Representatives: BEWAVE Zena, FIFTY FIFTY Sio, Billlie Sheon, Baby DONT Cry Beni, RESCENE Liv, QWER Hina, FIFTY FIFTY Hana, BADVILLAIN INA
- Contact sheet: `.cache/face_type_clusters/F03.jpg`

### F04 (44 members)

- Avg images/confidence: 8.6 / 0.602
- Top groups: NewJeans(4), MEOVV(3), VCHA(3), TWICE(3), KATSEYE(3)
- Representatives: MEOVV Anna, NewJeans Hanni, ITZY Yuna, I.MET.U Lea, NewJeans Haerin, TWICE Jeongyeon, ARISE Minju, ILLIT Minju
- Contact sheet: `.cache/face_type_clusters/F04.jpg`

### F05 (42 members)

- Avg images/confidence: 7.9 / 0.574
- Top groups: NiziU(3), RESCENE(3), LIGHTSUM(3), Hearts2Hearts(2), STAYC(2)
- Representatives: RESCENE May, Candy Shop Sui, MADEIN MiU, LIGHTSUM Nayoung, RESCENE Minami, LIMELIGHT MiU, RESCENE Zena, BEWAVE Jiun
- Contact sheet: `.cache/face_type_clusters/F05.jpg`

### F06 (41 members)

- Avg images/confidence: 8.4 / 0.551
- Top groups: Girls' Generation(4), EL7Z UP(2), LIGHTSUM(2), SECRET NUMBER(2), Red Velvet(2)
- Representatives: ICHILLIN' Sohee, H1-KEY Hwiseo, CLASS:y Hyungseo, LIGHTSUM Yujeong, EL7Z UP Nana, MOMOLAND Nayun, iii Soobin, EL7Z UP Hwiseo
- Contact sheet: `.cache/face_type_clusters/F06.jpg`

### F07 (35 members)

- Avg images/confidence: 7.1 / 0.607
- Top groups: Hearts2Hearts(5), tripleS(4), Baby DONT Cry(3), iii(3), NMIXX(2)
- Representatives: Hearts2Hearts Jiwoo, Baby DONT Cry Yihyun, tripleS Sullin, Hearts2Hearts Yuha, Baby DONT Cry Mia, ILLIT Yunah, NMIXX Jiwoo, SAY MY NAME Kanny
- Contact sheet: `.cache/face_type_clusters/F07.jpg`

### F08 (33 members)

- Avg images/confidence: 8.2 / 0.616
- Top groups: Lovelyz(3), SAY MY NAME(2), CLASS:y(2), KIIRAS(2), OH MY GIRL(2)
- Representatives: KIIRAS Kurumi, BADVILLAIN YunSeo, WJSN Luda, BABYMONSTER Rora, CLASS:y Chaewon, FIFTY FIFTY Saena, CLASS:y Hyeju, QWER Magenta
- Contact sheet: `.cache/face_type_clusters/F08.jpg`

### F09 (27 members)

- Avg images/confidence: 7.4 / 0.614
- Top groups: BADVILLAIN(2), EL7Z UP(2), SECRET NUMBER(2), WOOAH(2), ARTMS(1)
- Representatives: BADVILLAIN Vin, FIFTY FIFTY Yewon, CRAXY Wooah, MADEIN Suhye, BABYMONSTER Asa, LIMELIGHT Suhye, EL7Z UP Yuki, EL7Z UP Yeonhee
- Contact sheet: `.cache/face_type_clusters/F09.jpg`

### F10 (22 members)

- Avg images/confidence: 7.4 / 0.574
- Top groups: WJSN(3), ICHILLIN'(2), YOUNG POSSE(2), IVE(1), NMIXX(1)
- Representatives: NMIXX Bae, KIIRAS Kylie, ODD YOUTH Kanie, tripleS Kotone, Red Velvet Seulgi, ADYA Seowon, ARISE Jihu, SAY MY NAME Seungjoo
- Contact sheet: `.cache/face_type_clusters/F10.jpg`

## M Clusters

- Members: 687
- Clusters: 10
- Silhouette: 0.0287

### M01 (93 members)

- Avg images/confidence: 10.3 / 0.666
- Top groups: NCT(6), HORI7ON(5), BAE173(3), FANTASY BOYS(3), KickFlip(3)
- Representatives: TREASURE Junghwan, CRAVITY Allen, DRIPPIN Dongyun, CRAVITY Serim, XODIAC Gyumin, POW Hyunbin, xikers Junmin, AMPERS&ONE Mackiah
- Contact sheet: `.cache/face_type_clusters/M01.jpg`

### M02 (82 members)

- Avg images/confidence: 10.1 / 0.671
- Top groups: &TEAM(4), ENHYPEN(3), KickFlip(3), SEVENTEEN(3), xikers(3)
- Representatives: BOYNEXTDOOR Sungho, ENHYPEN Sunoo, Xdinary Heroes Jungsu, TWS Youngjae, &TEAM Jo, ENHYPEN Jay, &TEAM Harua, OMEGA X Jaehan
- Contact sheet: `.cache/face_type_clusters/M02.jpg`

### M03 (79 members)

- Avg images/confidence: 10.5 / 0.673
- Top groups: SEVENTEEN(5), GHOST9(3), idntt(3), NTX(3), XODIAC(3)
- Representatives: B.D.U Kim Minseo, TNX Sungjun, TNX Hwi, 82MAJOR Dogyun, SEVENTEEN Hoshi, WAKER Kohyeon, NTX Xiha, GHOST9 Shin
- Contact sheet: `.cache/face_type_clusters/M03.jpg`

### M04 (70 members)

- Avg images/confidence: 10.4 / 0.654
- Top groups: BE BOYS(3), IDID(3), idntt(3), YOUNITE(3), 8TURN(2)
- Representatives: idntt Park NuRi, YOUNITE DEY, POW Jungbin, BE BOYS Woncheon, TNX Junhyeok, Stray Kids Han, EVNNE Park Jihoo, AIMERS Yoel
- Contact sheet: `.cache/face_type_clusters/M04.jpg`

### M05 (68 members)

- Avg images/confidence: 9.9 / 0.645
- Top groups: idntt(3), POLARIX(3), TXT(3), AHOF(2), ALL(H)OURS(2)
- Representatives: POLARIX Shao Ziheng, n.SSign Hanjun, BAE173 J-Min, CLOSE YOUR EYES Jeon Minwook, BOYNEXTDOOR Jaehyun, POW Yorch, IDID Chu Yoochan, Stray Kids Hyunjin
- Contact sheet: `.cache/face_type_clusters/M05.jpg`

### M06 (63 members)

- Avg images/confidence: 10.5 / 0.649
- Top groups: 8TURN(3), VI'ENX(3), SF9(3), AHOF(2), 82MAJOR(2)
- Representatives: VI'ENX Younghoon, MCND Win, DAY6 Young K, NCHIVE HA.L, NINE.i Vari, EPEX Yewang, P1Harmony Intak, FANTASY BOYS Kang Minseo
- Contact sheet: `.cache/face_type_clusters/M06.jpg`

### M07 (63 members)

- Avg images/confidence: 10.9 / 0.671
- Top groups: CLOSE YOUR EYES(4), TREASURE(3), ZEROBASEONE(3), AMPERS&ONE(2), ARrC(2)
- Representatives: MCND Minjae, B.D.U Seunghun, CLOSE YOUR EYES Kenshin, BOYNEXTDOOR Riwoo, CLOSE YOUR EYES Song Seungho, CRAVITY Seongmin, CLOSE YOUR EYES Seo Kyoungbae, JUSTB Bain
- Contact sheet: `.cache/face_type_clusters/M07.jpg`

### M08 (62 members)

- Avg images/confidence: 10.4 / 0.658
- Top groups: DKB(3), SUPER JUNIOR(3), GOT7(3), CRAVITY(2), DXMON(2)
- Representatives: withus Junhyeok, ONE PACT Yedam, P1Harmony Theo, The Boyz Juyeon, CRAVITY Jungmo, ASTRO MJ, idntt Nam JiWoon, EPEX Jeff
- Contact sheet: `.cache/face_type_clusters/M08.jpg`

### M09 (59 members)

- Avg images/confidence: 10.2 / 0.659
- Top groups: NAZE(3), &TEAM(2), BTOB(2), EVNNE(2), FANTASY BOYS(2)
- Representatives: TWS Dohoon, EVNNE Mun Junghyun, AHOF Chih En, AND2BLE Han Yujin, NINE.i Jiho, NOWZ Jinhyuk, &TEAM K, BTOB Sungjae
- Contact sheet: `.cache/face_type_clusters/M09.jpg`

### M10 (48 members)

- Avg images/confidence: 9.9 / 0.661
- Top groups: NCT(3), DXMON(2), GHOST9(2), idntt(2), NINE.i(2)
- Representatives: ASTRO Cha Eunwoo, LUN8 Takuma, TEMPEST Eunchan, ALPHA DRIVE ONE Sanghyeon, TXT Soobin, YUHZ Junseong, YOUNITE Woono, The Wind Jang Hyounjoon
- Contact sheet: `.cache/face_type_clusters/M10.jpg`
