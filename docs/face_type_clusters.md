# Face Type Clustering Report

This is an exploratory clustering of idol face embeddings, not an objective taxonomy.
Known cross-group aliases are collapsed before clustering, and female/male members are clustered separately.

## Method

- Feature: normalized ArcFace face-recognition vector from aligned face crops
- Filter: member vectors with at least the requested minimum images/confidence
- Clustering: kmeans, cluster count selected near the target size by silhouette/balance
- Contact sheets: local representative crops nearest to each cluster centroid

## F Clusters

- Members: 392
- Clusters: 10
- Silhouette: 0.0297

### F01 (60 members)

- Avg images/confidence: 7.4 / 0.590
- Top groups: LIGHTSUM(4), NMIXX(3), Kep1er(3), NiziU(3), MADEIN(3)
- Representatives: NMIXX Jiwoo, NMIXX Sullyoon, LIGHTSUM Nayoung, RESCENE May, Baby DONT Cry Mia, EL7Z UP Yuki, IVE Liz, tripleS JooBin
- Contact sheet: `.cache/face_type_clusters/F01.jpg`

### F02 (49 members)

- Avg images/confidence: 7.4 / 0.611
- Top groups: USPEER(5), Kep1er(3), KiiiKiii(3), CRAXY(3), KIIRAS(3)
- Representatives: BADVILLAIN YunSeo, USPEER Seoyu, USPEER Chaena, ARISE Alisa, GFRIEND Yerin, Kep1er Chaehyun, CLASS:y Chaewon, KIIRAS Kurumi
- Contact sheet: `.cache/face_type_clusters/F02.jpg`

### F03 (49 members)

- Avg images/confidence: 7.7 / 0.589
- Top groups: FIFTY FIFTY(4), IVE(3), MEOVV(3), RESCENE(3), ILLIT(2)
- Representatives: FIFTY FIFTY Athena, RESCENE Minami, Baby DONT Cry Beni, RESCENE Liv, Candy Shop Soram, tripleS JiYeon, tripleS Xinyu, IVE Wonyoung
- Contact sheet: `.cache/face_type_clusters/F03.jpg`

### F04 (43 members)

- Avg images/confidence: 7.1 / 0.581
- Top groups: WJSN(3), ARTMS(2), Hearts2Hearts(2), ADYA(2), PRIKIL(2)
- Representatives: ADYA Seowon, WJSN Seola, Hearts2Hearts Jiwoo, tripleS Kotone, MADEIN Suhye, LIMELIGHT Suhye, Hearts2Hearts Juun, ARISE Jihu
- Contact sheet: `.cache/face_type_clusters/F04.jpg`

### F05 (41 members)

- Avg images/confidence: 8.3 / 0.589
- Top groups: VCHA(3), TWICE(3), LE SSERAFIM(2), Hearts2Hearts(2), XG(2)
- Representatives: NewJeans Hanni, I.MET.U Lea, FIFTY FIFTY Hana, BEWAVE Jiun, ITZY Lia, UDTT Kwon Yejin, ITZY Yuna, AtHeart Nahyun
- Contact sheet: `.cache/face_type_clusters/F05.jpg`

### F06 (36 members)

- Avg images/confidence: 8.1 / 0.602
- Top groups: WJSN(4), Queenz Eye(3), KISS OF LIFE(2), QWER(2), BADVILLAIN(2)
- Representatives: QWER Hina, BEWAVE Zena, BADVILLAIN INA, Billlie Sheon, WJSN Luda, Billlie Suhyeon, Candy Shop Yuina, CLASS:y Hyeju
- Contact sheet: `.cache/face_type_clusters/F06.jpg`

### F07 (33 members)

- Avg images/confidence: 8.2 / 0.599
- Top groups: NiziU(2), XG(2), YOUNG POSSE(2), Girls' Generation(2), aespa(1)
- Representatives: ARTMS JinSoul, ADYA Yeonsu, ITZY Yeji, FIFTY FIFTY Keena, aespa Karina, XG Juria, Red Velvet Wendy, KIIRAS LingLing
- Contact sheet: `.cache/face_type_clusters/F07.jpg`

### F08 (29 members)

- Avg images/confidence: 8.0 / 0.609
- Top groups: NewJeans(3), VVUP(3), izna(2), BABYMONSTER(2), ICHILLIN'(2)
- Representatives: Hearts2Hearts Stella, ICHILLIN' Joonie, UNCHILD Yeeun, NewJeans Haerin, ICHILLIN' Chaerin, BABYMONSTER Ruka, UNCHILD Tina, H1-KEY Yel
- Contact sheet: `.cache/face_type_clusters/F08.jpg`

### F09 (27 members)

- Avg images/confidence: 8.6 / 0.572
- Top groups: LE SSERAFIM(2), Red Velvet(2), OH MY GIRL(2), ALLDAY PROJECT(2), aespa(1)
- Representatives: LE SSERAFIM Kim Chaewon, IU IU, STAYC Sumin, Candy Shop Sui, ALLDAY PROJECT Youngseo, CLASS:y Hyungseo, H1-KEY Seoi, LE SSERAFIM Kazuha
- Contact sheet: `.cache/face_type_clusters/F09.jpg`

### F10 (25 members)

- Avg images/confidence: 8.3 / 0.570
- Top groups: SECRET NUMBER(4), Lovelyz(3), ICHILLIN'(2), Girls' Generation(2), MOMOLAND(2)
- Representatives: SECRET NUMBER Min C, SECRET NUMBER Minji, OH MY GIRL Jiho, BLACKPINK Rosé, MOMOLAND Nayun, ICHILLIN' Sohee, STAYC Sieun, ICHILLIN' Jackie
- Contact sheet: `.cache/face_type_clusters/F10.jpg`

## M Clusters

- Members: 688
- Clusters: 11
- Silhouette: 0.0307

### M01 (85 members)

- Avg images/confidence: 10.2 / 0.650
- Top groups: &TEAM(4), XLOV(3), BAE173(2), BLITZERS(2), BTOB(2)
- Representatives: VIXX N, ENHYPEN Sunoo, WAKER Kohyeon, EVNNE Park Hanbin, idntt Nam JiWoon, NCT Jungwoo, ASTRO MJ, Xdinary Heroes Jungsu
- Contact sheet: `.cache/face_type_clusters/M01.jpg`

### M02 (75 members)

- Avg images/confidence: 10.3 / 0.657
- Top groups: CLOSE YOUR EYES(4), AHOF(3), YUHZ(3), ALL(H)OURS(2), ALPHA DRIVE ONE(2)
- Representatives: CLOSE YOUR EYES Seo Kyoungbae, CLOSE YOUR EYES Song Seungho, CRAVITY Seongmin, MCND Win, TNX Junhyeok, NCT Jaemin, NCHIVE HA.L, CRAVITY Minhee
- Contact sheet: `.cache/face_type_clusters/M02.jpg`

### M03 (71 members)

- Avg images/confidence: 10.6 / 0.653
- Top groups: Pentagon(5), Hi-Fi Un!corn(3), IDID(3), idntt(3), Stray Kids(3)
- Representatives: idntt Park NuRi, idntt Lee KyuHyuk, Stray Kids Han, DRIPPIN Hyeop, POW Jungbin, TNX Hwi, YOUNITE DEY, TNX Taehun
- Contact sheet: `.cache/face_type_clusters/M03.jpg`

### M04 (70 members)

- Avg images/confidence: 11.1 / 0.674
- Top groups: TREASURE(5), ALPHA DRIVE ONE(3), EPEX(3), KickFlip(3), NCT(3)
- Representatives: Stray Kids Seungmin, GHOST9 Shin, DRIPPIN Dongyun, CRAVITY Allen, TEMPEST Taerae, xikers Yechan, ONF Hyojin, ALPHA DRIVE ONE Xinlong
- Contact sheet: `.cache/face_type_clusters/M04.jpg`

### M05 (64 members)

- Avg images/confidence: 10.9 / 0.658
- Top groups: JUSTB(3), MODYSSEY(3), xikers(3), BLITZERS(2), BOYNEXTDOOR(2)
- Representatives: BOYNEXTDOOR Riwoo, TEMPEST Hyeongseop, KEYVITUP Hyunmin, AND2BLE Zhang Hao, BOYNEXTDOOR Jaehyun, JUSTB DY, VIXX Hyuk, CRAVITY Wonjin
- Contact sheet: `.cache/face_type_clusters/M05.jpg`

### M06 (63 members)

- Avg images/confidence: 9.9 / 0.675
- Top groups: XODIAC(4), HORI7ON(3), NCT(3), OMEGA X(3), CRAVITY(2)
- Representatives: GHOST9 Woojin, EPEX Jeff, XODIAC Leo, KickFlip Donghwa, SEVENTEEN Jun, ARrC Rioto, ALPHA DRIVE ONE Anxin, ENHYPEN Jay
- Contact sheet: `.cache/face_type_clusters/M06.jpg`

### M07 (55 members)

- Avg images/confidence: 10.3 / 0.655
- Top groups: TAN(3), BTS(3), ALL(H)OURS(2), AMPERS&ONE(2), E'LAST(2)
- Representatives: The Boyz Younghoon, AND2BLE Han Yujin, WEi Daehyeon, xikers Jinsik, MONSTA X Minhyuk, WHIB Leejeong, NOWZ Jinhyuk, 82MAJOR Seongbin
- Contact sheet: `.cache/face_type_clusters/M07.jpg`

### M08 (54 members)

- Avg images/confidence: 9.9 / 0.641
- Top groups: SEVENTEEN(3), WINNER(3), Dragon Pony(2), NTX(2), OMEGA X(2)
- Representatives: TNX Sungjun, AIMERS Seunghwan, BOYNEXTDOOR Woonhak, SEVENTEEN Wonwoo, DAYCHILD Siwoo, 82MAJOR Seongmo, MONSTA X Kihyun, WINNER Seunghoon
- Contact sheet: `.cache/face_type_clusters/M08.jpg`

### M09 (53 members)

- Avg images/confidence: 9.7 / 0.674
- Top groups: CORTIS(4), &TEAM(3), FANTASY BOYS(3), NCT(2), POLARIX(2)
- Representatives: FANTASY BOYS Lee Hanbin, FANTASY BOYS Ling Qi, TWS Youngjae, SUPER JUNIOR Ryeowook, TREASURE Junghwan, NOWZ Yeonwoo, LUN8 Takuma, &TEAM Maki
- Contact sheet: `.cache/face_type_clusters/M09.jpg`

### M10 (51 members)

- Avg images/confidence: 10.4 / 0.669
- Top groups: idntt(3), AHOF(2), AND2BLE(2), BOYNEXTDOOR(2), CLOSE YOUR EYES(2)
- Representatives: YUHZ Jaeil, BAE173 Youngseo, FANTASY BOYS Hikari, Hi-Fi Un!corn Hyunyul, idntt Choi TaeIn, The Wind Kim Heesoo, OMEGA X Xen, The Boyz Juyeon
- Contact sheet: `.cache/face_type_clusters/M10.jpg`

### M11 (47 members)

- Avg images/confidence: 9.6 / 0.674
- Top groups: DXMON(2), FANTASY BOYS(2), HORI7ON(2), MCND(2), NCT(2)
- Representatives: The Wind Jang Hyounjoon, VAY ONN Peng Jinyu, VAYONN Peng Jinyu, n.SSign Sungyun, TEMPEST Eunchan, GHOST9 Jinwoo, TREASURE Haruto, TRENDZ Yechan
- Contact sheet: `.cache/face_type_clusters/M11.jpg`
