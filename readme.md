# Team VCL (Victory Congratulation Lab)
* 팀장: 이예림
* 팀원: 고준규 고현성 이연경 이진우

***

# 프로젝트 소개
최근 YouTube 의 영상들에서 더보기 란에 있는 타임스탬프들을 쉽게 찾아볼 수 있다. <br>
영상 제작자가 남겨둔 타임스탬프는 제작자의 의도에 맞게 영상의 컨텐츠를 나눌 수 있는 단위로 쓰이고 있다. <br>
그렇다면 수많은 영상이 있는 상황에서, 비슷한 종류의 타임스탬프만 모아서 본다면 영상들의 흐름을 더 쉽게 파악할 수 있을 것이다. <br>
<br>
이 아이디어는 실제로 "그것이 알고 싶다"와 같은 시사 프로그램에서 자주 쓰이고 있다. <br>
이러한 프로그램에서는 긴 흐름을 가지는 사건이 어떻게 흘러갔는지 쉽게 파악하기 위해서 많은 뉴스 영상들의 앵커, 기자, 인터뷰 등의 구간을 교차편집하여 사용하곤 한다. <br>
하지만 이렇게 몇 년에 걸친 많은 뉴스 영상들에서 특정 구간만을 추출하는 것은 아직까지 사람의 몫으로 남아 있는 문제다. <br>
<br>
본 프로젝트는 어떤 사건에 대해 관련된 뉴스 영상들을 나열하고, 해당 영상들의 앵커, 기자, 인터뷰, 자료 화면 구간의 시작 지점을 자동으로 추출하는 딥러닝 기반의 모델 개발을 목적으로 한다. <br>
이 프로그램을 사용하여 특정 인물/사건 테마 별 플레이리스트 작성 혹은 이를 이용한 팩트 체크 등과 같은 2차 콘텐츠 제작에 들어가는 수고를 크게 줄일 수 있을 것을 기대한다.

***

# 데모 페이지
<https://leeyell.github.io/hackathon2020/demo/>

***

# Model
C.Feichtenhofer et al. <br>
__SlowFast Networks for Video Recognition__. ICCV, 2019 <br>
![network](/pic/network.png)

SlowFast 모델은 이전 논문들과는 다르게 Optical Flow 를 사용하지 않는다. <br>
하지만 모델의 두 Path 중 하나인 Slow Path 에서, Frame 시퀀스를 보는 stride 를 작게 주어 연속적으로 보기 때문에 Optical Flow 와 동일한 효과를 볼 수 있다고 해당 논문의 저자들은 주장하고 있다.

***

# Dataset

`AI학습용 구축 데이터 - 보도_MBN종합뉴스` 비디오 데이터 활용 

## Dataset Labeling
> 1. 앵커멘트 <br>
> ![앵커](/pic/앵커.jpg)
> 2. 기자멘트 <br>
> ![기자](/pic/기자.jpg)
> 3. 인터뷰 <br>
> ![인터뷰](/pic/interview.jpg)
> 4. 자료화면 <br>
>![자료화면](/pic/screen.jpg)

## Dataset 구축 방법
각 뉴스 영상에서 Label 이 변하는 구간에 대해 타임 스탬프 형식의 GT 를 정의. <br>
영상의 Frame 은 16 FPS 로 추출하였음. <br>
Data Sample 1개는 16 Frame 의 RGB Image 시퀀스와 해당 시퀀스의 Label 인덱스로 이루어져 있다.

***

# keyword 추출
Demo 페이지에서의 시연을 위해 각 뉴스 영상의 Title 에서 검색어가 될 키워드를 추출하였다. <br>
> * `MBN종합뉴스-552-2019.xlsx` 파일의 title column을 'konlpy'의 형태소분석기 'Hannanum'을 사용하여 명사 추출
> * Data의 title 중 명사가 제대로 추출되지 않았다면 추가하지 않음 *ex) '...보아하니'*
> * 검색의 속도 향상을 위해 Inverted Index 구조를 적용하여 *{key:keyword, value:해당 keyword를 갖는 뉴스의 id}* 형태의 Dictionary 로 정의
> * keyword를 입력했을 때, 해당 keyword를 포함하고 있는 뉴스 영상을 검색할 수 있다.