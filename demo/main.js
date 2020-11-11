var timepoint_class_names = ['앵커 멘트', '기자 멘트', '인터뷰', '자료 화면'];
var data_path = 'data/data.json';
var keyword_data_path = 'data/raw/keyword2id.json'
var data, keyword_data, data_json, keyword_data_json;

var reader = new XMLHttpRequest() || new ActiveXObject('MSXML2.XMLHTTP');

function custom_select () {
    var x, i, j, l, ll, selElmnt, a, b, c;
    /* Look for any elements with the class "custom-select": */
    x = document.getElementsByClassName("custom-select");
    l = x.length;
    for (i = 0; i < l; i++) {
    selElmnt = x[i].getElementsByTagName("select")[0];
    ll = selElmnt.length;
    /* For each element, create a new DIV that will act as the selected item: */
    a = document.createElement("DIV");
    a.setAttribute("class", "select-selected");
    a.innerHTML = selElmnt.options[selElmnt.selectedIndex].innerHTML;
    x[i].appendChild(a);
    /* For each element, create a new DIV that will contain the option list: */
    b = document.createElement("DIV");
    b.setAttribute("class", "select-items select-hide");
    for (j = 1; j < ll; j++) {
        /* For each option in the original select element,
        create a new DIV that will act as an option item: */
        c = document.createElement("DIV");
        c.innerHTML = selElmnt.options[j].innerHTML;
        c.addEventListener("click", function(e) {
            /* When an item is clicked, update the original select box,
            and the selected item: */
            var y, i, k, s, h, sl, yl;
            s = this.parentNode.parentNode.getElementsByTagName("select")[0];
            sl = s.length;
            h = this.parentNode.previousSibling;
            for (i = 0; i < sl; i++) {
            if (s.options[i].innerHTML == this.innerHTML) {
                s.selectedIndex = i;
                h.innerHTML = this.innerHTML;

                /*------ item click -------*/
                // console.log(this.innerHTML); 
                set_option(i);

                y = this.parentNode.getElementsByClassName("same-as-selected");
                yl = y.length;
                for (k = 0; k < yl; k++) {
                y[k].removeAttribute("class");
                }
                this.setAttribute("class", "same-as-selected");
                break;
            }
            }
            h.click();
        });
        b.appendChild(c);
    }
    x[i].appendChild(b);
    a.addEventListener("click", function(e) {
        /* When the select box is clicked, close any other select boxes,
        and open/close the current select box: */
        e.stopPropagation();
        closeAllSelect(this);
        this.nextSibling.classList.toggle("select-hide");
        this.classList.toggle("select-arrow-active");
    });
    }
}

function closeAllSelect(elmnt) {
    /* A function that will close all select boxes in the document,
    except the current select box: */
    var x, y, i, xl, yl, arrNo = [];
    x = document.getElementsByClassName("select-items");
    y = document.getElementsByClassName("select-selected");
    xl = x.length;
    yl = y.length;
    for (i = 0; i < yl; i++) {
      if (elmnt == y[i]) {
        arrNo.push(i)
      } else {
        y[i].classList.remove("select-arrow-active");
      }
    }
    for (i = 0; i < xl; i++) {
      if (arrNo.indexOf(i)) {
        x[i].classList.add("select-hide");
      }
    }
  }

/*------------------data loader-------------------------*/
function loadFile(path) {
    reader.open('get', path, true); 
    reader.onreadystatechange = get_data;
    reader.send(null);
}
function get_data() {
    if(reader.readyState==4) {
        // console.log(reder.responseText);
        // updateval = reader.responseText;
    }
}
/*------------------------------------------------------*/



// var data = '{"850508": {"video_ytid": "GucT4pGzk4A", "date": "20190101", "cut_points": "00:00,00:12,00:30,00:34,00:52,00:58,01:15,01:23,", "point_class": "0,1,2,1,2,1,2,1,", "video_length": 95}, "850711": {"video_ytid": "n_nr-E7uuDM", "date": "20190102", "cut_points": "00:00,00:18,01:15,01:26,", "point_class": "0,1,2,1,", "video_length": 108}, "851000": {"video_ytid": "BtZSNOFJYEY", "date": "20190103", "cut_points": "00:00,00:17,00:32,00:44,00:55,01:10,01:16,", "point_class": "0,1,2,1,2,1,3,", "video_length": 96}, "851261": {"video_ytid": "md2n43n77M4", "date": "20190104", "cut_points": "00:00,00:16,00:34,00:38,01:03,", "point_class": "0,1,2,1,3,", "video_length": 98}, "851506": {"video_ytid": "6U0wW4LAbjI", "date": "20190105", "cut_points": "00:00,00:19,00:46,00:54,", "point_class": "0,1,2,1,", "video_length": 88}, "851630": {"video_ytid": "WURFNzXAexk", "date": "20190106", "cut_points": "00:00,00:25,00:44,01:09,01:18,01:27,01:38,", "point_class": "0,1,3,2,1,2,1,", "video_length": 108}, "851884": {"video_ytid": "PRTFQfROVkU", "date": "20190107", "cut_points": "00:00,00:22,00:47,", "point_class": "0,1,3,", "video_length": 115}, "852135": {"video_ytid": "_wtvpy7SSmo", "date": "20190108", "cut_points": "00:00,00:27,00:49,01:08,01:18,01:22,", "point_class": "0,1,3,3,2,1,", "video_length": 113}, "852404": {"video_ytid": "jeT_iyDVoVA", "date": "20190109", "cut_points": "00:00,00:16,00:40,00:50,01:11,01:25,01:35,01:44,", "point_class": "0,1,2,1,2,1,2,1,", "video_length": 116}, "852627": {"video_ytid": "CFx9F7ZOBQs", "date": "20190110", "cut_points": "00:00,00:18,00:45,01:02,01:11,01:23,", "point_class": "0,1,2,1,2,1,", "video_length": 94}, "852898": {"video_ytid": "1La0hoTSS1E", "date": "20190111", "cut_points": "00:00,00:16,00:38,00:55,01:11,01:24,01:34,01:45,", "point_class": "0,1,3,3,2,1,2,1,", "video_length": 120}, "853127": {"video_ytid": "-r1ib25Vf9c", "date": "20190112", "cut_points": "00:00,00:16,00:31,00:57,01:24,", "point_class": "0,1,3,3,3,", "video_length": 106}, "853223": {"video_ytid": "e5lRgwTMLug", "date": "20190113", "cut_points": "00:00,00:14,00:25,00:50,01:02,01:11,01:21,01:35,", "point_class": "0,1,3,2,1,2,1,3,", "video_length": 112}, "853461": {"video_ytid": "NXgSZuzie-M", "date": "20190114", "cut_points": "00:00,00:24,00:41,00:57,01:08,01:13,01:29,01:37,", "point_class": "0,1,2,1,2,1,2,1,", "video_length": 121}, "853724": {"video_ytid": "_DDdRPhmkBA", "date": "20190115", "cut_points": "00:00,00:15,00:34,00:48,01:00,01:19,01:26,", "point_class": "0,1,3,2,3,2,1,", "video_length": 94}, "853982": {"video_ytid": "-0e6oc1IeIw", "date": "20190116", "cut_points": "00:00,00:17,00:46,01:11,01:30,", "point_class": "0,1,3,2,1,", "video_length": 99}, "854247": {"video_ytid": "o5xYSznX9JI", "date": "20190117", "cut_points": "00:00,00:20,00:44,00:59,01:12,01:27,01:32,", "point_class": "0,1,2,1,2,1,3,", "video_length": 116} }'
// var keyword_data = '{    "60년": ["850508"],    "황금돼지": ["850508"],    "4대": ["850711"],    "그룹": ["850711"],    "총수": ["850711"],    "4대그룹": ["850711"],    "4대그룹총수": ["850711"],    "그룹총수": ["850711"],    "한자리에": ["850711"],    "한국당": ["850711"],    "경련": ["850711"],    "불참": ["850711"],    "김태우": ["851000", "851506", "850534"],    "참고인": ["851000"],    "조사": ["851000", "851506", "851008"],    "청와대": ["851000", "854247", "854825", "870385", "870721", "891164"],    "비밀누설": ["851000", "850534"],    "신재민": ["851261"],    "이틀째": ["851261"],    "병원": ["851261"],    "두문불출": ["851261"],    "친구들": ["851261"],    "호소문": ["851261"],    "발표": ["851261"],    "연이틀": ["851506"],    "진실": ["851506"],    "반품": ["851630"],    "납품업체": ["851630"],    "직원": ["851630", "851008"],    "마트": ["851630"],    "단독": ["851884", "852135", "855575", "890461"],    "환경부": ["851884"],    "임원추천위": ["851884"],    "의결": ["851884"],    "2번": ["851884"],    "묵살": ["851884"],    "후임": ["851884"],    "대통령": ["851884", "852627", "853724", "871762", "891533", "851612", "857454"],    "문대통령": ["851884", "852627", "853724", "871762", "891533", "851612"],    "문재인대통령": ["851884", "852627", "853724", "871762", "891533", "851612"],    "환경특보": ["851884"],    "박병대": ["852135"],    "임종헌": ["852135"],    "자리": ["852135"],    "마련": ["852135"],    "고문료": ["852135"],    "수천만": ["852135"],    "나경원": ["852404", "854247"],    "조국": ["852404", "850534"],    "영혼": ["852404"],    "탈곡기": ["852404"],    "조해주": ["852404"],    "청문회": ["852404", "889965"],    "파행": ["852404"],    "노동계": ["852627"],    "자세": ["852627"],    "성폭력": ["852898", "855059"],    "운동선수": ["852898"],    "73명": ["852898"],    "초등학생": ["852898"],    "피해": ["852898"],    "올해": ["853127", "853205"],    "북": ["853127", "853223", "853724", "850510", "851637"],    "비핵화": ["853127", "850510"],    "상당": ["853127"],    "전": ["853127"],    "기대": ["853127", "856109"],    "인도적": ["853127"],    "지원": ["853127", "857454"],    "대북제재": ["853127"],    "완화": ["853127"],    "5.18": ["853223"],    "북한군": ["853223", "853724"],    "개입": ["853223"],    "지만원": ["853223"],    "탈북민": ["853223"],    "고소당해": ["853223"],    "고교": ["853461"],    "때": ["853461"],    "코치": ["853461"],    "성폭행": ["853461"],    "전직": ["853461"],    "유도": ["853461"],    "선수": ["853461"],    "폭로": ["853461"],    "문재인": ["851884", "852627", "853724", "871762", "891533", "851612"],    "정부": ["853724", "891309", "892034"],    "문재인정부": ["851884", "852627", "853724", "871762", "891533"],    "첫": ["853724", "872049", "850510", "851046", "852433", "853205"],    "국방백서": ["853724"],    "북한": ["853127", "853223", "853724", "891309", "850510", "851637"],    "적": ["853724"],    "8년": ["853724"],    "삭제": ["853724"],    "구도": ["853982"],    "땅값": ["853982"],    "오름세": ["853982"],    "손혜원": ["853982", "854494", "855309"],    "목포": ["853982", "855309"],    "부동산": ["853982"],    "투기": ["853982"],    "의혹": ["853982", "870704", "889965", "851008", "854508"],    "부동산투기": ["853982"],    "부동산투기의혹": ["853982"],    "김정숙": ["854247"],    "여사": ["854247"],    "김정숙여사": ["854247"],    "절친": ["854247"],    "예의": ["854247"],    "반박동영상": ["854494"],    "동영상": ["854494"],    "공개": ["854494"],    "절대": ["854494"],    "스웨덴": ["854723"],    "방문": ["854723", "891164"],    "남북미": ["854723"],    "3자": ["854723"],    "실무협상": ["854723"],    "주목": ["854723"],    "집값": ["854825"],    "대책": ["854825"],    "집값대책": ["854825"],    "빙상계": ["855059"],    "6건": ["855059"],    "확인": ["855059"],    "전명규": ["855059"],    "교수": ["855059", "890461"],    "은폐": ["855059"],    "기자회견": ["855309", "851612"],    "가짜뉴스": ["855309"],    "해명": ["855309"],    "월급": ["855575"],    "대낮": ["855575"],    "난동": ["855575"],    "남성": ["855575", "891303"],    "한순간": ["855861"],    "잿더미": ["855861"],    "울산": ["855861"],    "농수산물시장": ["855861"],    "큰불": ["855861"],    "주차장": ["856131"],    "살": ["856131"],    "피의자": ["856131"],    "징역": ["856131"],    "30년": ["856131"],    "딸": ["856131"],    "재범": ["856131"],    "오늘": ["870141"],    "52": ["870141"],    "52시간": ["870141", "889971"],    "주52시간": ["870141", "889971"],    "근무": ["870141"],    "처벌": ["870141"],    "다급": ["870145"],    "경제부총리": ["870145"],    "국회": ["870145"],    "탄력근로제": ["870145"],    "법안": ["870145"],    "절실": ["870145"],    "바나나": ["870149"],    "비닐": ["870149", "872786"],    "사용": ["870149", "872786"],    "비닐사용": ["870149", "872786"],    "비닐봉지": ["870149"],    "금지": ["870149"],    "첫날": ["870149"],    "혼선": ["870149"],    "이재영": ["870163"],    "통합": ["870163"],    "달성": ["870163"],    "시대": ["870163", "870727"],    "3명": ["870385"],    "청문": ["870385"],    "청문보고": ["870385"],    "요청": ["870385"],    "임명": ["870385", "871762"],    "강행": ["870385"],    "윤중천": ["870409"],    "조력자": ["870409"]}'


function load_data() {
    // load data
    reader.open('get', data_path, true);
    reader.onreadystatechange = function(){
        if (reader.readyState == 4) {
            console.log('data load');
            data = reader.responseText;
            data_json = JSON.parse(data);

            // load keyword data
            reader.open('get', keyword_data_path, true);
            reader.onreadystatechange = function(){
                console.log('keyword data load');
                if (reader.readyState == 4) {
                    keyword_data = reader.responseText;
                    keyword_data_json = JSON.parse(keyword_data);
                    // console.log(keyword_data);
                }
            }
            reader.send(null);
        }
    }
    reader.send(null);

    
}

function submit(text) {
    // clear right wrapper
    var right_wrapper = document.getElementById('right_wrapper');
    right_wrapper.innerHTML = '<div id="timeline_bar"></div>'

    vids = keyword_data_json[text];
    for (var i = 0; i < vids.length; i++) {
        // create_video_ui('VAScKwUvn4s', '2019년 7월 1일', '00:00,00:20,00:43,00:58,01:08,01:28,01:42', '0,1,3,2,1,2,1', 115);
        var cur_data = data_json[vids[i]];
        create_video_ui(cur_data['video_ytid'], cur_data['date'], cur_data['cut_points'], cur_data['point_class'], cur_data['video_length']);
    }
}




/*------- set option with data -------*/
function set_option(idx) {
    // set video timeline options
    var video_elems = document.getElementsByClassName('timeline_element');
    for (var elem of video_elems) {
        var timeline = elem.querySelector('.video_timeline');
        var _cut_points = timeline.getAttribute('points');
        var _point_classes = timeline.getAttribute('point_class');

        var cut_point_split = _cut_points.split(',');
        var point_classes_split = _point_classes.split(',');

        timeline.innerHTML = "";        
        
        for (var i = 0; i < cut_point_split.length; i++) {
            var cls_idx = point_classes_split[i];
            if (cls_idx.length == 0) {
                continue;
            }
            if (cls_idx == idx-1) {
                var time = cut_point_split[i];
                var time_minute = parseInt(time.substring(0,2));
                var time_second = parseInt(time.substring(3,5));
                create_cut_point_ui(elem, idx-1, time_minute*60+time_second);
            }
        }
    }
}
 
/*------- create video UI -------*/

function create_video_ui(id, date, cut_points, point_classes, video_length) {
    var wrapper = document.getElementById('right_wrapper');
    
    var element = document.createElement('div');
    element.classList.toggle('timeline_element');

    var timeline_dot = document.createElement('div');
    timeline_dot.classList.toggle('timeline_dot');

    var video_date = document.createElement('div');
    video_date.classList.toggle('video_date');
    date_str = date.substring(0,4)+ "년 " + date.substring(4,6) + "월 " + date.substring(6,8) + "일";
    video_date.innerHTML = date_str;

    var ytplayer = document.createElement('iframe');
    ytplayer.classList.toggle('ytplayer');
    var src = "https://www.youtube.com/embed/" + id + "?autoplay=1&controls=0"
    ytplayer.setAttribute('src', src);
    ytplayer.setAttribute('frameborder','0');

    var video_timeline = document.createElement('div');
    video_timeline.classList.toggle('video_timeline');
    video_timeline.setAttribute('points', cut_points);
    video_timeline.setAttribute('point_class', point_classes);
    video_timeline.setAttribute('video_length', video_length);
    video_timeline.setAttribute('video_id', id);


    wrapper.appendChild(element);
    element.appendChild(timeline_dot);
    element.appendChild(video_date);
    element.appendChild(ytplayer);
    element.appendChild(video_timeline);
}

function create_cut_point_ui(elem, cls_idx, time) {
    var timeline = elem.querySelector('.video_timeline');
    var video_length = timeline.getAttribute('video_length');
    var time_point = document.createElement('div');
    var time_des = document.createElement('div');

    time_des.innerHTML = timepoint_class_names[cls_idx];

    time_point.classList.toggle('time_point');
    time_des.classList.toggle('time_des');

    time_point.style.left = time / video_length * 100 + "%";

    time_point.appendChild(time_des);
    timeline.appendChild(time_point);

    time_point.addEventListener('click', ()=> {
        var ytplayer = elem.querySelector('.ytplayer');
        var id = timeline.getAttribute('video_id');
        var src = "https://www.youtube.com/embed/" + id + "?autoplay=1&controls=0&start=" + time;
        ytplayer.setAttribute('src', src);
    });
}

// create_video_ui('VAScKwUvn4s', '2019년 7월 1일', '00:00,00:20,00:43,00:58,01:08,01:28,01:42', '0,1,3,2,1,2,1', 115);
// create_video_ui('h2oT2T9NiZw', '2019년 7월 1일', '00:00,00:18,00:49,00:56,01:00,01:24,01:30', '0,1,2,1,3,2,1', 112);

/*------- search -------*/
function init() {
    load_data();
    custom_select();

    search_form = document.getElementById('form_keyword');
    search_form.addEventListener('keyup', (e)=> {
        if (e.keyCode === 13) submit(search_form.value);
    });

    /* If the user clicks anywhere outside the select box,
    then close all select boxes: */
    document.addEventListener("click", closeAllSelect);
}

init();










