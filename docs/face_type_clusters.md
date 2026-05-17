# Face Type Clustering Report

This is an exploratory clustering of idol face embeddings, not an objective taxonomy.
Known cross-group aliases are collapsed before clustering, and female/male members are clustered separately.

## Method

- Feature: normalized ArcFace face-recognition vector from aligned face crops
- Filter: member vectors with at least the requested minimum images/confidence
- Clustering: kmeans, cluster count selected near the target size by silhouette/balance
- Contact sheets: local representative crops nearest to each cluster centroid

## F Clusters

- Members: 389
- Clusters: 10
- Silhouette: 0.0304

### F01 (52 members)

- Avg images/confidence: 7.1 / 0.606
- Top groups: tripleS(4), NewJeans(3), MEOVV(3), iii(3), MIMIIROSE(3)
- Representatives: UNCHILD Tina, iii Ahra, BABYMONSTER Ruka, USPEER Seoyu, ICHILLIN' Joonie, Hearts2Hearts Stella, tripleS JiYeon, Billlie Sheon
- Contact sheet: `.cache/face_type_clusters/F01.jpg`

### F02 (49 members)

- Avg images/confidence: 7.7 / 0.579
- Top groups: SECRET NUMBER(6), Lovelyz(4), BADVILLAIN(3), H1-KEY(2), SAY MY NAME(2)
- Representatives: STAYC Sumin, LIMELIGHT Suhye, MADEIN Suhye, H1-KEY Hwiseo, iii Soobin, VVUP Kim, aespa Giselle, EL7Z UP Hwiseo
- Contact sheet: `.cache/face_type_clusters/F02.jpg`

### F03 (47 members)

- Avg images/confidence: 8.2 / 0.597
- Top groups: Kep1er(4), Hearts2Hearts(3), Baby DONT Cry(3), LIGHTSUM(3), tripleS(3)
- Representatives: ODD YOUTH Myah, NMIXX Jiwoo, Baby DONT Cry Mia, LIGHTSUM Nayoung, STAYC J, Hearts2Hearts Jiwoo, tripleS JooBin, Hearts2Hearts A-na
- Contact sheet: `.cache/face_type_clusters/F03.jpg`

### F04 (45 members)

- Avg images/confidence: 7.8 / 0.610
- Top groups: CLASS:y(3), QWER(2), KIIRAS(2), tripleS(2), USPEER(2)
- Representatives: CLASS:y Chaewon, BADVILLAIN YunSeo, WJSN Luda, KIIRAS Kurumi, ARISE Alisa, BABYMONSTER Rora, USPEER Daon, CLASS:y Hyeju
- Contact sheet: `.cache/face_type_clusters/F04.jpg`

### F05 (44 members)

- Avg images/confidence: 8.3 / 0.591
- Top groups: NiziU(3), Queenz Eye(3), Kep1er(2), XG(2), KIIRAS(2)
- Representatives: Kep1er Yujin, ICHILLIN' Chaerin, ITZY Yeji, ADYA Yeonsu, aespa Karina, Kep1er Chaehyun, NiziU Mako, SAY MY NAME Seungjoo
- Contact sheet: `.cache/face_type_clusters/F05.jpg`

### F06 (39 members)

- Avg images/confidence: 8.6 / 0.581
- Top groups: Hearts2Hearts(2), NiziU(2), QWER(2), XG(2), AtHeart(2)
- Representatives: NewJeans Hanni, I.MET.U Lea, FIFTY FIFTY Hana, UDTT Kwon Yejin, ITZY Yuna, SAY MY NAME Shuie, ITZY Lia, BEWAVE Jiun
- Contact sheet: `.cache/face_type_clusters/F06.jpg`

### F07 (38 members)

- Avg images/confidence: 7.8 / 0.597
- Top groups: FIFTY FIFTY(4), IVE(3), TWICE(3), LE SSERAFIM(2), RESCENE(2)
- Representatives: Baby DONT Cry Beni, FIFTY FIFTY Athena, FIFTY FIFTY Yewon, NMIXX Sullyoon, RESCENE Minami, FIFTY FIFTY Sio, Candy Shop Soram, BEWAVE Zena
- Contact sheet: `.cache/face_type_clusters/F07.jpg`

### F08 (30 members)

- Avg images/confidence: 7.4 / 0.598
- Top groups: WJSN(3), EL7Z UP(2), ICHILLIN'(2), Lovelyz(2), IVE(1)
- Representatives: LIGHTSUM Juhyeon, BEWAVE Gowoon, STAYC Sieun, IVE Gaeul, KIIRAS Kylie, MADEIN MiU, NiziU Miihi, LIMELIGHT MiU
- Contact sheet: `.cache/face_type_clusters/F08.jpg`

### F09 (23 members)

- Avg images/confidence: 7.2 / 0.588
- Top groups: ADYA(3), CRAXY(2), Lovelyz(2), WJSN(2), ARTMS(1)
- Representatives: Lovelyz Mijoo, BADVILLAIN Vin, ARTMS Kim Lip, ADYA Seowon, Rocking doll Juri, CRAXY Wooah, CLASS:y Hyungseo, ADYA Chaeeun
- Contact sheet: `.cache/face_type_clusters/F09.jpg`

### F10 (22 members)

- Avg images/confidence: 8.2 / 0.555
- Top groups: NiziU(2), RESCENE(2), UDTT(2), Red Velvet(2), BLACKPINK(2)
- Representatives: RESCENE May, GFRIEND Yuju, BLACKPINK Jennie, RESCENE Woni, MAMAMOO Moonbyul, Red Velvet Joy, PRIKIL Nana, TWICE Nayeon
- Contact sheet: `.cache/face_type_clusters/F10.jpg`

## M Clusters

- Members: 686
- Clusters: 10
- Silhouette: 0.0308

### M01 (84 members)

- Avg images/confidence: 11.2 / 0.666
- Top groups: DKB(3), E'LAST(3), NINE.i(3), AHOF(2), ALPHA DRIVE ONE(2)
- Representatives: NCHIVE HA.L, MCND Win, EPEX Yewang, CRAVITY Seongmin, EVNNE Park Jihoo, B.D.U Seunghun, NXD Jaemin, YUHZ Jaeil
- Contact sheet: `.cache/face_type_clusters/M01.jpg`

### M02 (83 members)

- Avg images/confidence: 10.4 / 0.669
- Top groups: CRAVITY(4), XODIAC(3), BLITZERS(2), ENHYPEN(2), NCT(2)
- Representatives: ONE PACT Yedam, idntt Nam JiWoon, EPEX Jeff, ALPHA DRIVE ONE Anxin, CRAVITY Jungmo, hrtz.wav Riaan, EVNNE Park Hanbin, ASTRO MJ
- Contact sheet: `.cache/face_type_clusters/M02.jpg`

### M03 (71 members)

- Avg images/confidence: 10.1 / 0.659
- Top groups: idntt(5), The KingDom(3), ARrC(2), DAYCHILD(2), DKB(2)
- Representatives: POW Jungbin, DAYCHILD Eden, TNX Sungjun, YOUNITE DEY, DRIPPIN Hyeop, The KingDom Dann, AIMERS Yoel, B.D.U Kim Minseo
- Contact sheet: `.cache/face_type_clusters/M03.jpg`

### M04 (71 members)

- Avg images/confidence: 10.6 / 0.661
- Top groups: idntt(3), P1Harmony(3), TAN(3), Stray Kids(3), B.A.P(3)
- Representatives: idntt Lee KyuHyuk, idntt Park NuRi, P1Harmony Theo, Stray Kids Han, 8TURN Kyungmin, NOWZ Yeonwoo, NouerA Gihyeon, ZEROBASEONE Seok Matthew
- Contact sheet: `.cache/face_type_clusters/M04.jpg`

### M05 (70 members)

- Avg images/confidence: 10.3 / 0.673
- Top groups: ALPHA DRIVE ONE(3), GHOST9(3), NCT(3), SEVENTEEN(3), TREASURE(3)
- Representatives: 82MAJOR Dogyun, ALPHA DRIVE ONE Xinlong, EPEX Wish, GHOST9 Shin, ONF Hyojin, SEVENTEEN Dino, TREASURE Yoshi, NCT WISH Yushi
- Contact sheet: `.cache/face_type_clusters/M05.jpg`

### M06 (69 members)

- Avg images/confidence: 10.3 / 0.649
- Top groups: &TEAM(3), BOYNEXTDOOR(3), ENHYPEN(3), Pentagon(3), Hi-Fi Un!corn(2)
- Representatives: BOYNEXTDOOR Sungho, ENHYPEN Sunoo, TNX Hwi, Xdinary Heroes Jungsu, TEMPEST Hyeongseop, WAKER Kohyeon, ENHYPEN Jay, LUN8 Chael
- Contact sheet: `.cache/face_type_clusters/M06.jpg`

### M07 (66 members)

- Avg images/confidence: 10.1 / 0.678
- Top groups: FANTASY BOYS(4), AHOF(2), BAE173(2), DXMON(2), HORI7ON(2)
- Representatives: VAY ONN Peng Jinyu, VAYONN Peng Jinyu, YUHZ Junseong, GHOST9 Jinwoo, n.SSign Sungyun, The Wind Jang Hyounjoon, FANTASY BOYS Hong Sungmin, FANTASY BOYS Kim Gyurae
- Contact sheet: `.cache/face_type_clusters/M07.jpg`

### M08 (65 members)

- Avg images/confidence: 10.6 / 0.665
- Top groups: NCT(5), HORI7ON(4), ARrC(3), XODIAC(3), AMPERS&ONE(2)
- Representatives: TREASURE Junghwan, CRAVITY Allen, CRAVITY Serim, DRIPPIN Dongyun, XODIAC Gyumin, n.SSign Laurence, TREASURE Asahi, Stray Kids Seungmin
- Contact sheet: `.cache/face_type_clusters/M08.jpg`

### M09 (64 members)

- Avg images/confidence: 10.0 / 0.633
- Top groups: MONSTA X(3), BTS(3), AHOF(2), AND2BLE(2), &TEAM(2)
- Representatives: The Boyz Younghoon, CLOSE YOUR EYES Song Seungho, AND2BLE Han Yujin, VI'ENX Younghoon, 82MAJOR Seongbin, MONSTA X Minhyuk, IDID Baek Junhyuk, NOWZ Jinhyuk
- Contact sheet: `.cache/face_type_clusters/M09.jpg`

### M10 (43 members)

- Avg images/confidence: 8.5 / 0.652
- Top groups: POLARIX(3), XLOV(3), CORTIS(2), MODYSSEY(2), NCT(2)
- Representatives: POLARIX Zai, VAY ONN Sun Jiayang, Stray Kids Felix, TXT Yeonjun, VAYONN Sun Jiayang, POLARIX Shao Ziheng, CLOSE YOUR EYES Jeon Minwook, ALPHA DRIVE ONE Arno
- Contact sheet: `.cache/face_type_clusters/M10.jpg`
