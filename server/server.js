
var express = require('express');
var app = express();
var server = require('http').Server(app);
var io = require('socket.io').listen(server);
var mysql = require('mysql');
const PATH = require('path');
const bodyParser = require('body-parser');
const STATS = require('./config.js');

let connected_players = {}; //한번이라도 연결되었던 플레이어의 데이터를 저장하는 객체
let chat_list = {}; // 현재 채팅방에 접속중인 플레이어의 데이터를 저장하는 객체

const STATIC_PATH = PATH.join(__dirname, '../client');

app.use('/css',express.static(__dirname + '/css'));
app.use('/js',express.static(__dirname + '../js'));
app.use('/client',express.static(STATIC_PATH));
app.use('/External',express.static(__dirname + '/External'));
app.use('/assets',express.static(__dirname + '/assets'));
app.use('/node_modules',express.static(__dirname + '/node_modules'));

//노드서버에서 POST 로 전달받은 데이터를 처리해주는 미들웨어 모듈
app.use(bodyParser.urlencoded({extended: true}));
app.use(bodyParser.json());

//mysql
const gameserver = mysql.createPool({
   user: 'root',
   password: '123',
   database: 'gamedb',
});

/*//아파치서버를 통해 회원가입된 데이터베이스를 연결해주는 부분
var memberserver = mysql.createConnection({
    user: 'root',
    password: '123',
    database: 'members'
});*/


//mysql 서버 접속해서 데이터 베이스의 특정 테이블 목록 출
/*gameserver.query('SELECT * FROM character_data', function(error, result, fields) {
    if(error) {
        console.log('쿼리 문장에 오류가 있습니다.');
    } else {
        console.log(result);
    }
});

memberserver.query('SELECT * FROM member', function(error, result, fields) {
    if(error) {
        console.log('쿼리 문장에 오류가 있습니다.');
    } else {
        console.log(result.length);
    }
});*/


/*app.get('/',function(req,res){
    res.sendFile(STATIC_PATH+'/index.html');
});*/

//body parser을 통해 접속해야 하므로, 로그인을 안하면 주소창에 주소를 쳐도 접속할수가 없다.
let idd;
app.post('/', function (req, res) { //req에는 이전 페이지에서  post로 전달한 id값이 담겨있다.
    console.log("로그인한 아이디 : " , req.body.id);
    idd = req.body.id;

    //mysql의 gamedb에 저장된 아이디가 이미 있는지 확인
    gameserver.query('SELECT * FROM character_data where id=?', idd , function(err, rows, fields) {
        if(err) { throw err; }

        if(rows.length === 1){

            console.log("이미 있는 아이디 그러므로 저장을 안한다!");
            connected_players[idd] = rows; // 한번이라도 연결되었던 플레이어 데이터를 저장하는 객체에 정보 저장
            console.log("저장된 데이터\n", connected_players, chat_list);
            res.sendFile(STATIC_PATH+'/index.html');
            //gameserver.connection.release();

        }else {
            console.log("처음접속했습니다!!!");
            // 저장된 아이디가 없으면 저장을 해준다. 아이디는 유니크하도록 설정되어있다.
            gameserver.query('insert into character_data (id) values ("' + idd + '")', function(err, rows) {
                if(err) { throw err; }
                console.log("Data inserted!", rows);
                gameserver.query('SELECT * FROM character_data where id=?', idd , function(err, rows, fields) {
                    connected_players[idd] = rows;
                    console.log("저장된 데이터\n", connected_players, idd);
                });
                    res.sendFile(STATIC_PATH+'/index.html');
                //gameserver.release();

            });
        }
    });
});


server.lastPlayderID = 0;

server.listen(process.env.PORT || 8081,function(){ // 8081 포트로 접속이 가능하다.
    console.log('Listening on '+server.address().port);
});


let GameBasicData = {
    worldSizeX: 1200,
    worldSizeY: 1200,
    grid: null,
    ObstaclesNumber: 30,
    ObstaclesLocation: null
};

let CollectableNumber =  20;
let CollectableLocation = [];



//console.log("first !! " + GameBasicData.ObstaclesNumber);

//맵 생성, 생성된 데이터를 각 클라이언트에게 전달
initMap(GameBasicData.worldSizeX, GameBasicData.worldSizeY);
//console.log("second !! " + GameBasicData.grid);

generateObstacleLocation(GameBasicData.ObstaclesNumber); // 위치좌표 생성
generateCollectablesLocation(CollectableNumber); // 위치좌표 생성


function initMap(wx, wy){
    GameBasicData.grid = [];
    var gridSize = 32;
    var gridsX = Math.floor(wx / gridSize);
    var gridsY = Math.floor(wy / gridSize);
    console.log("gridsX : " + gridsX + " gridsY : " + gridsY);
    for (var x = 0; x < gridsX; x++) {
        for (var y = 0; y < gridsY; y++) {
            var gridX = x * gridSize;
            var gridY = y * gridSize;
            GameBasicData.grid.push({x:gridX, y:gridY});
        }
    }
    shuffle(GameBasicData.grid);

}

function shuffle(array) {
    var currentIndex = array.length, temporaryValue, randomIndex ;

    // While there remain elements to shuffle...
    while (0 !== currentIndex) {

        // Pick a remaining element...
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;

        // And swap it with the current element.
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
    }

    return array;
}
function generateObstacleLocation(number) {
    GameBasicData.ObstaclesLocation = [];
    for (var i = 0; i < number; i++) {
        GameBasicData.ObstaclesLocation.push(getRandomLocation());
    }
}

function generateCollectablesLocation(number) {
    CollectableLocation = [];
    for (var i = 0; i < number; i++) {
        CollectableLocation.push(getRandomLocation());
    }
}

function getRandomLocation() {

    var gridIndex = 0;
    if(GameBasicData.grid[gridIndex] === undefined || GameBasicData.grid[gridIndex] === undefined){
        return {location:{x: 0, y: 0}, spriteIndex: 0};
    }
    var x = GameBasicData.grid[gridIndex].x;
    var y = GameBasicData.grid[gridIndex].y;
    var spriteIndex = Math.floor(Math.random() * 10);

    GameBasicData.grid.splice(gridIndex, 1);
    gridIndex++;
    if (gridIndex === GameBasicData.grid.length) {
        shuffle(GameBasicData.grid);
        gridIndex = 0;
    }
    return {location:{x, y}, spriteIndex};
}

const players = {}; // 현재 접속중인 캐릭터의 정보를 담고있는 객체

/*setInterval(function () {
    //일정시간 동안 반응이 없으면 mysql서버와의 연결이 끊기기 때문에
//일정간격으로 쿼리를 날려주는 부분이다.
    // 5초마다 쿼리를 날리며 접속을 유지하고, 회원수 상태를 띄움
    gameserver.query('SELECT 1');
    //console.log("접속중인 회원 수 : " + Object.keys(players).length + " 접속했던 회원 수 : " + Object.keys(connected_players).length + "\n", connected_players);
}, 5000);*/

/*setInterval(function () {
    generateCollectablesLocation(Math.floor(Math.random() * CollectableNumber));
    console.log("Generate Collectable!! : " + CollectableLocation.length);
    io.emit('generate_collectables', CollectableLocation);
}, 10000);*/

setTimeout(function () {
    setInterval(function () {
        generateCollectablesLocation(Math.floor(Math.random() * (CollectableNumber-10) + 10));

        console.log("Generate Collectable!! : " + CollectableLocation.length);
        io.emit('generate_collectables', CollectableLocation);


    }, 30000);
}, 10000);

setTimeout(function () {
    setInterval(function () {

        let tip =
            ['자신보다 높은 등급의 상대를 죽이면 더 높은 점수를 획득할 수 있습니다.',
            '죽으면 점수가 깎입니다.',
            '워리어는 체력이 엄청 높습니다.',
            '닌자는 공격속도는 엄청 빠르지만, 공격사거리가 매우 짧습니다.',
            '등급 구성 : [ 브론드 -> 실버 -> 골드 -> 플래티넘 -> 다이아 -> 마스터 -> 그랜드마스터 -> 신 ]', //5
            '승급은 100포인트 단위로 이루어 집니다. 점수가 낮아지면 등급도 낮아집니다.',
            '모든 포션은 효과가 중복적용이 안됩니다.',
            '체력회복 포션은 체력이 가득찬 경우 먹어도 소용없습니다.',
            '즐거운 시간 되세요!!!!',
            '가끔 게임이 이상하면 다시 재접속 해주세요..', //10
            '상자는 30초마다 젠됩니다.',
            '메이지는 공격력이 엄청 쎄지만, 체력이 매우 낮습니다.',
            '레인저는 직업중 가장 사거리가 깁니다. 하지만 공격속도가 가장 느립니다.',
            '힐러는 현재 가장 쓸모가 없습니다.',
            '게임 플레이 중 엔터키를 누르면 채팅메세지를 보낼 수 있습니다.',
            '이 메세지는 5초마다 나옵니다.',
            '모든 정보(점수, 죽인수, 죽은횟수)는 자동저장됩니다.',
            '직업별 스킬이 추가될 예정입니다.']; //14

        //console.log(Math.floor(Math.random() * 11));
        io.emit('broadcast_game_info', '[TIP] '+tip[Math.floor(Math.random() * tip.length)], 'tip');
    }, 5000);
}, 10000);



io.on('connection', socket => {


    chat_list[socket.id] = idd;
    if(connected_players[idd] !== undefined) {
        connected_players[idd].socketid = socket.id; // 연결되었던 플레이어 리스트에 해당 아이디에 접속한 소켓아이디값을 넣어준다.
    }
    console.log("연결된 플레이어 : ", chat_list, connected_players[idd]);

    io.to(socket.id).emit('character-config', STATS); // 정보창에 캐릭터 정보 주는 부분
    io.to(socket.id).emit('update-character-info', connected_players[chat_list[socket.id]]); // 정보창 업데이트 부분

    //console.log(" TEST : " ,connected_players[idd][0].Kill );

    socket.on('logindata', data =>{
        /*if(data !== null){ // 로그인데이터가 있다는 소리는 죽어서 재시작한 경우
            console.log("restart login id : " + data, socket.id);
            var tmp = {id: data, socket: socket.id};
            io.to(socket.id).emit('loginID', tmp);
            io.to(socket.id).emit('update-character-info', connected_players[chat_list[socket.id]]);


        }else {*/
            console.log("login id : " + chat_list[socket.id], socket.id, "player number : ", players);
            let tmp = {id: chat_list[socket.id], socket: socket.id};
            //chat_list[socket.id] = idd;
            //console.log("저장된 데이터2\n", connected_players);
            io.to(socket.id).emit('loginID', tmp, STATS);
            io.to(socket.id).emit('update-character-info', connected_players[chat_list[socket.id]]);

       // }

    });

    socket.on('new-game', () => {
        console.log('GameBasicData');

        //처음 게임에 접속하면 게임에 대한 기본 정보(맵크기, 개체수 정보 등)과 클래스들의 속성을 전달해준다.
        let Data = { GameBasicData: GameBasicData, CLASS_STATS: STATS , collectable: CollectableLocation};
        io.to(socket.id).emit('generate-world', Data);
    });

    socket.on('new-player', state => {
        console.log('New player joined with state:\n', state , '\n-----------------------------------\n');
        //console.log('palyerbasicdata:\n', playerData);
        //connected_players[state.playerName.name].socketid = socket.id;

        players[state.playerName.name] = state;
        players[state.playerName.name].tierImage.grade = connected_players[state.playerName.name][0].Grade;
        players[state.playerName.name].damageBuff = false;
        players[state.playerName.name].speedBuff = false;
        players[state.playerName.name].warrior_skill = false; // 무적 사용시 true
        players[state.playerName.name].warrior_cool_time = false; // 무적 사용시 true
        players[state.playerName.name].ninja_skill = false; // 투명 사용시 true
        players[state.playerName.name].ninja_cool_time = false; // 닌자스킬 쿨타임, 스킬 쿨타임이 돌아가는 중이면 트루, 끝나면 폴스
        players[state.playerName.name].healer_skill = false; //
        players[state.playerName.name].heal_effect = false; //
        players[state.playerName.name].heal_value = 0; //
        players[state.playerName.name].ranger_cool_time = false; //
        players[state.playerName.name].mage_cool_time = false; //

        players[state.playerName.name].attack_damage = STATS[players[state.playerName.name].type].ATTACK_POWER;
        players[state.playerName.name].speed.value = STATS[players[state.playerName.name].type].SPEED;
        //players[state.playerName.name].MaxHP = STATS[players[state.playerName.name].type].HP;

        // Emit the update-players method in the client side
        io.emit('update-players', players);
        io.to(socket.id).emit('update_hp', players[state.playerName.name].HP.val, players[state.playerName.name].MaxHP.val);
        console.log("현재 접속중인 플레이어 수 : " + Object.keys(players).length, players);

    });

    socket.on('move-player', data => {
        //console.log('move player  : ' + socket.id);

        const { x, y, angle, type, attack_damage, attack_speed, attack_range, playerName, tierImage, speed, HP} = data;

        // If the player is invalid, return
        if (players[data.playerName.name] === undefined) {
            return
        }

        // Update the player's data if he moved
        players[data.playerName.name].x = x;
        players[data.playerName.name].y = y;
        players[data.playerName.name].angle = angle;
        players[data.playerName.name].type = type;
        //players[data.playerName.name].attack_damage = attack_damage;
        players[data.playerName.name].attack_speed = attack_speed;
        players[data.playerName.name].attack_range = attack_range;
        players[data.playerName.name].playerName = {
            name: playerName.name,
            x: playerName.x,
            y: playerName.y
        };

        players[data.playerName.name].tierImage = {
            grade: tierImage.grade,
            x: tierImage.x,
            y: tierImage.y
        };

        players[data.playerName.name].speed = {
            value: speed.value,
            x: speed.x,
            y: speed.y
        };

        players[data.playerName.name].HP = {
            val: HP.val
        };
        // Send the data back to the client
        //console.log("Move Player Update", players);
        io.emit('update-players', players);
    });

    socket.on('attack-player', function(data){
        //console.log("receive attack!!");

        // console.log('attack to '+socket.id+', '+data.x + " / " + data.y);

        if(players[data.name] === undefined){
            //console.log('attack to return');
            return
        }
        let sendingdata = {
            sendingid : data.name,
            point: {
                worldX:data.x,
                worldY:data.y,
            }
        };

        io.emit('update-attack',sendingdata);
    });

    socket.on('ranger-skill-attack-player', function(data){
        //console.log("receive attack!!");

        // console.log('attack to '+socket.id+', '+data.x + " / " + data.y);

        if(players[data.name] === undefined){
            //console.log('attack to return');
            return
        }
        let sendingdata = {
            sendingid : data.name,
            point: {
                worldX:data.x,
                worldY:data.y,
            }
        };

        io.emit('update-ranger-skill-attack',sendingdata);
    });

    /*socket.on('damage-player-to-otherplayer', function (data) {
        console.log("receive damage : " + data.attackID + " / " + data.damagedID);

        if(players[socket.id] === undefined){
            return
        }

        if(data.attackID === socket.id){
            players[data.damagedID].HP.val -= 10;
            console.log("check : " + players[data.damagedID].playerName.name + " / " + players[data.damagedID].HP.val);

            io.emit('update-players', players);
        }

    });*/

    socket.on('damage-player-to', function (data) {
        //console.log("attack ", players);
        console.log("공격 : [ " + data.attackID + " ] -> [ " + data.damagedID + " ] " + players[data.attackID].attack_damage);

        if(players[data.attackID] === undefined || players[data.damagedID] === undefined){
            return
        }


        /* 티어점수
        0 ~ 100 : 브론즈(0)        101 ~ 200 : 실버(1)        201 ~ 300 : 골드(2)        301 ~ 400 : 플래티넘(3)
        401 ~ 500 : 다이아몬드(4)   501 ~ 600 : 마스터(5)       601 ~ 700 : 그랜드마스터(6)  701 ~ 800 : 신(7)
        *
        * 같은 등급의 상대를 죽인 경우 : 점수 + 10
        * 나보다 높은 등급의 상대를 죽인 경우 : 점수 + 30
        * 나보다 낮은 등급의 상대를 죽인 경우 : 점수 + 5
        * 죽은 경우 : 점수 - 3
        *
        * */


        if( players[data.damagedID].warrior_skill === false){
            if(data.skill === 'ranger'){ // 레인저 스킬에 맞았을때 처리하는 부분
                let tmp_speed = players[data.damagedID].speed.value;
                console.log("변하기전 스피드 : " + tmp_speed);
                players[data.damagedID].speed.value = 0; // 스피드를 0으로 줘서 이동속도를 감소시킨다.
                io.to(connected_players[data.damagedID].socketid).emit('broadcast_game_info', '얼음화살에 맞아 3초 동안 이동이 불가능해집니다.', 'skill');

                setTimeout(descreaseDM(), 3*1000);
                function descreaseDM() {
                    return function(){
                        if( players[data.damagedID] === undefined){
                            return
                        }else {
                            console.log("이동속도 감소 해제" + players[data.damagedID].speed.value);
                            players[data.damagedID].speed.value = STATS[players[data.damagedID].type].SPEED;
                            io.emit('update-players', players);
                            console.log("이동속도 감소 해제2" + players[data.damagedID].speed.value);

                            io.to(connected_players[data.damagedID].socketid).emit('broadcast_game_info', '이동속도 감소 지속시간이 끝났습니다.', 'skill');
                        }
                    };
                }
            }else if(data.skill === 'mage'){
                let damage = Math.floor(players[data.attackID].attack_damage/3);
                players[data.damagedID].HP.val -= damage;
                let sendText = data.attackID + "님이 " +  data.damagedID+ "님에게 광역공격으로 " +damage + "만큼의 데미지를 주었습니다!!";
                io.emit('broadcast_game_info', sendText, 'skill');
            } else {
                players[data.damagedID].HP.val -= players[data.attackID].attack_damage;
                let sendText = data.attackID + "님이 " +  data.damagedID+ "님에게 " +players[data.attackID].attack_damage + "만큼의 데미지를 주었습니다!!";
                io.emit('broadcast_game_info', sendText, 'fight');
            }
        }


        io.to(connected_players[data.damagedID].socketid).emit('update_hp', players[data.damagedID].HP.val, players[data.damagedID].MaxHP.val);


            //캐릭터가 죽었을때 클라이언트에게 콜백보내는 부분
        if(players[data.damagedID].HP.val <= 0){

            /*
            *  전투 결과의 전적 업데이트 부분 : 점수를 올리거나 낮추고, 킬수와 죽은수를 업데이트 해준다.
            * */
            connected_players[data.attackID][0].Kill++;
            connected_players[data.damagedID][0].Died++;

            // 공격자의 점수를 증가시킨다. 상대방의 등급에 따라 증가폭이 다르다.
            if(connected_players[data.attackID][0].Grade < connected_players[data.damagedID][0].Grade){ // 내가 죽인 상대가 나보다 등급이 높은 경우
                connected_players[data.attackID][0].Point += 30;
            }else if(connected_players[data.attackID][0].Grade > connected_players[data.damagedID][0].Grade){//내가 죽인 상대가 나보다 등급이 낮은 경우
                connected_players[data.attackID][0].Point += 5;
            } else { // 그 외의 경우 (등급이 같은 경우)
                connected_players[data.attackID][0].Point += 10;
            }

            // 죽은 사람의 점수를 깍는다.
            connected_players[data.damagedID][0].Point -= 3;
            if(connected_players[data.damagedID][0].Point < 0){
                connected_players[data.damagedID][0].Point = 0;
            }
            // ===================================================


            connected_players[data.attackID][0].Grade = TierGrade(connected_players[data.attackID][0].Point);
            connected_players[data.damagedID][0].Grade = TierGrade(connected_players[data.damagedID][0].Point);
            players[data.attackID].tierImage.grade = TierGrade(connected_players[data.attackID][0].Point);
            players[data.damagedID].tierImage.grade = TierGrade(connected_players[data.damagedID][0].Point);

            let sendText = data.damagedID + "님이 " + data.attackID + "님에게 죽었습니다.";
            io.emit('broadcast_game_info', sendText, 'fight');
            console.log(data.damagedID + " 님이  " + data.attackID + "님에게 죽었습니다.", connected_players[data.attackID][0].Kill);

            //데이터베이스 저장 부분 ======================================================================================
            //공격자의 데이터 저장 부분 = 저장데이터: 킬수, 점수, 등급
            let sql = 'UPDATE character_data t SET t.Kill = ?, t.Point = ?, t.Grade = ? WHERE t.idx = ?';
            let params = [connected_players[data.attackID][0].Kill, connected_players[data.attackID][0].Point
                , connected_players[data.attackID][0].Grade, connected_players[data.attackID][0].idx];

            gameserver.query(sql, params , function (err, rows, fields) {
                if(err){
                    console.log(err);
                }else {
                    console.log(data.attackID + "의 전적이 업데이트 되었습니다.");
                }
            });

            //죽은 사람의 데이터 저장 부분 = 저장데이터 : 죽은수, 점수, 등급
            sql = 'UPDATE character_data t SET t.Died = ?, t.Point = ?, t.Grade = ? WHERE t.idx = ?';
            params = [connected_players[data.damagedID][0].Died, connected_players[data.damagedID][0].Point
                , connected_players[data.damagedID][0].Grade, connected_players[data.damagedID][0].idx];

            gameserver.query(sql, params , function (err, rows, fields) {
                if(err){
                    console.log(err);
                }else {
                    console.log(data.damagedID + "의 전적이 업데이트 되었습니다.");
                }
            });


            //=========================================================================================================
            // 접속한 클라이언트에게 자신의 데이터를 보내 정보창을 업데이트 해준다.
            io.to(socket.id).emit('update-character-info', connected_players[chat_list[socket.id]]);


            let tempdata = players[data.damagedID];
            io.emit('died-player', tempdata, data.attackID);
            delete players[data.damagedID];

            console.log("DiedPlayer update", players);

        }
        //console.log("check2 : " + players[data.damagedID].playerName.name + " / " + players[data.damagedID].HP.val);
        io.emit('update-players', players);


    });

    socket.on("update_box", function (data, position, spriteIndex) {
        CollectableLocation = data;
        io.emit('generate_collectables', CollectableLocation, position, spriteIndex);
    });

    //아이템 처리 관련 IO ==============================================================================================

    //힐
    socket.on('heal', function (ID, heal_value) {
        if(players[ID].MaxHP.val === players[ID].HP.val){
            io.to(connected_players[ID].socketid).emit('broadcast_game_info', '이미 체력이 가득차있습니다. 포션을 마셔도 효과가 없습니다.', 'potion');

        }else {
            players[ID].HP.val += heal_value;
            if(players[ID].MaxHP.val < players[ID].HP.val){
                players[ID].HP.val = players[ID].MaxHP.val;
                io.to(connected_players[ID].socketid).emit('broadcast_game_info', '체력회복 포션을 마십니다. 체력이 완전히 회복되었습니다.', 'potion');

            }else {
                io.to(connected_players[ID].socketid).emit('broadcast_game_info', '체력회복 포션을 마십니다. 체력이'+ heal_value+ '만큼 회복합니다!!', 'potion');
            }

            io.emit('update-players', players);
            io.to(connected_players[ID].socketid).emit('update_hp', players[ID].HP.val, players[ID].MaxHP.val);
        }


    });

    //체력증가
    socket.on('increaseHP', function (ID, increase_value) {
        if(players[ID].MaxHP.buff === false){ // 이미 체력증가 버프를 받고 있다면, 중복적용되지 않도록 하기위함
            players[ID].MaxHP.val += increase_value;
            players[ID].MaxHP.buff = true;

            let buff_time = 10;
            let count = buff_time;

            io.to(connected_players[ID].socketid).emit('decreaseHP_cool_time', count, 'start');
            setInterval(function () {

                count--;
                io.to(connected_players[ID].socketid).emit('decreaseHP_cool_time', count, 'during');

                if( count <= 0){
                    clearInterval(this);
                    io.to(connected_players[ID].socketid).emit('decreaseHP_cool_time', count, 'stop');
                }

            }, 1000);

            setTimeout(descreaseHP(increase_value), buff_time*1000); // bufftime초 후에 버프 종료
            function descreaseHP(val) {
                var temp = val;
                return function(){
                    console.log("다시감소" + temp);

                    players[ID].MaxHP.val -= temp;
                    players[ID].MaxHP.buff = false;
                    if( players[ID].HP.val > players[ID].MaxHP.val){
                        players[ID].HP.val = players[ID].MaxHP.val;
                    }

                        io.emit('update-players', players);
                    io.to(connected_players[ID].socketid).emit('broadcast_game_info', '체력증가 포션의 지속시간이 끝났습니다!!', 'potion');
                    io.to(connected_players[ID].socketid).emit('update_hp', players[ID].HP.val, players[ID].MaxHP.val);

                };
            }
            io.emit('update-players', players);
            io.to(connected_players[ID].socketid).emit('broadcast_game_info', '체력증가 포션을 마십니다. 체력이'+increase_value+ '만큼 증가합니다!!', 'potion');
            io.to(connected_players[ID].socketid).emit('update_hp', players[ID].HP.val, players[ID].MaxHP.val);
        }
    });

    // 공격력 증가
    socket.on('increaseDAMAGE', function (ID, increase_value) {
        if(players[ID].damageBuff === false){ // 이미 공격력증가 버프를 받고 있다면, 중복적용되지 않도록 하기위함
            players[ID].attack_damage += increase_value;
            players[ID].damageBuff = true;

            let buff_time = 10;
            let count = buff_time;

            io.to(connected_players[ID].socketid).emit('decreaseDAMAGE_cool_time', count, 'start');
            setInterval(function () {

                count--;
                io.to(connected_players[ID].socketid).emit('decreaseDAMAGE_cool_time', count, 'during');

                if( count <= 0){
                    clearInterval(this);
                    io.to(connected_players[ID].socketid).emit('decreaseDAMAGE_cool_time', count, 'stop');
                }

            }, 1000);

            setTimeout(descreaseDM(increase_value), buff_time*1000); // 버프시간후에 버프 종료
            function descreaseDM(val) {
                var temp = val;
                return function(){
                    console.log("공격력 다시감소" + temp);

                    players[ID].attack_damage -= temp;
                    players[ID].damageBuff = false;

                    io.emit('update-players', players);
                    io.to(connected_players[ID].socketid).emit('broadcast_game_info', '공격력증가 포션의 지속시간이 끝났습니다!!', 'potion');

                };
            }
            io.emit('update-players', players);
            io.to(connected_players[ID].socketid).emit('broadcast_game_info', '공격력증가 포션을 마십니다. 공격력이'+increase_value+ '만큼 증가합니다!!', 'potion');
        }
    });

    //스피드증가
    socket.on('increaseSPEED', function (ID, increase_value) {
        if(players[ID].speedBuff === false){ // 이미 스피드증가 버프를 받고 있다면, 중복적용되지 않도록 하기위함
            console.log("스피드 증가전 : "+players[ID].speed.value);
            players[ID].speed.value += increase_value;
            console.log("스피드 증가 : "+players[ID].speed.value);
            players[ID].speedBuff = true;

            let buff_time = 10;
            let count = buff_time;

            io.to(connected_players[ID].socketid).emit('decreaseSPEED_cool_time', count, 'start');
            setInterval(function () {

                count--;
                io.to(connected_players[ID].socketid).emit('decreaseSPEED_cool_time', count, 'during');

                if( count <= 0){
                    clearInterval(this);
                    io.to(connected_players[ID].socketid).emit('decreaseSPEED_cool_time', count, 'stop');
                }

            }, 1000);

            setTimeout(descreaseDM(increase_value), buff_time*1000);
            function descreaseDM(val) {
                var temp = val;
                return function(){
                    console.log("스피드 다시감소" + temp);

                    players[ID].speed.value -= temp;
                    players[ID].speedBuff = false;

                    io.emit('update-players', players);
                    io.to(connected_players[ID].socketid).emit('broadcast_game_info', '스피드포션의 지속시간이 끝났습니다!!', 'potion');

                };
            }
            io.emit('update-players', players);
            io.to(connected_players[ID].socketid).emit('broadcast_game_info', '스피드포션을 마십니다. 스피드가'+increase_value+ '만큼 증가합니다!!', 'potion');
        }
    });
    //=================================================================================================================
/*
*  각 직업별 스킬
*  워리어 - 5초간 무적 ok 10초 쿨타임
*  메이지 - 광역 범위 공격
*  레인저 - 얼음화살
*  힐러 - 셀프힐링 ok
*  닌자 - 5초간 투명 ok 10초 쿨타임
*
* */


    socket.on('warrior_skill', function (ID) {
        if(players[ID].warrior_skill === false && players[ID].warrior_cool_time === false){ // 이미 스피드증가버프를 받고 있다면, 중복적용되지 않도록 하기위함
            console.log("워리어스킬 사용");
            players[ID].warrior_skill = true;
            players[ID].warrior_cool_time = true;

            let buff_time = 5;
            let count = 10;

            io.to(connected_players[ID].socketid).emit('warrior_skill_using', count, 'start');
            setInterval(function () {

                count--;
                io.to(connected_players[ID].socketid).emit('warrior_skill_using', count, 'during');

                if( count <= 0){
                    clearInterval(this);
                    if(players[ID] !== undefined){
                        players[ID].warrior_cool_time = false;
                        io.to(connected_players[ID].socketid).emit('warrior_skill_using', count, 'stop');
                        io.to(connected_players[ID].socketid).emit('broadcast_game_info', '무적스킬의 쿨타임이 회복되었습니다.', 'skill');

                    }
                }

            }, 1000);

            setTimeout(descreaseDM(), buff_time*1000);
            function descreaseDM() {
                return function(){
                    console.log("워리어 무적 해제");
                    if( players[ID] === undefined){
                        console.log("undefined!!");
                        return;
                    }else {
                        console.log("끝!!!!");

                        players[ID].warrior_skill = false;

                        io.emit('update-players', players);
                        io.to(connected_players[ID].socketid).emit('broadcast_game_info', '무적스킬의 지속시간이 끝났습니다.', 'skill');
                    }


                };
            }
            io.emit('update-players', players);
            io.to(connected_players[ID].socketid).emit('broadcast_game_info', '무적스킬을 사용하였습니다.' + buff_time + '초 동안 체력이 감소하지 않습니다.', 'skill');
        }
    });

    socket.on('mage_skill', function (ID) {
        if(players[ID].mage_cool_time === false) { //
            console.log("메이지스킬 사용");
            players[ID].mage_cool_time = true;

            let count = 1; // 메이지 스킬 쿨타임 5초

            io.to(connected_players[ID].socketid).emit('mage_skill_using', count, 'start');
            setInterval(function () {

                count--;
                io.to(connected_players[ID].socketid).emit('mage_skill_using', count, 'during');

                if (count <= 0) {
                    clearInterval(this);
                    if (players[ID] !== undefined) {
                        players[ID].mage_cool_time = false;
                        io.to(connected_players[ID].socketid).emit('mage_skill_using', count, 'stop');
                        io.to(connected_players[ID].socketid).emit('broadcast_game_info', '광역공격 쿨타임이 회복되었습니다.', 'skill');

                    }
                }
            }, 1000);

            io.emit('update-players', players);
            io.to(connected_players[ID].socketid).emit('broadcast_game_info', '광역공격 스킬을 사용하였습니다.', 'skill');
            io.emit('update-mage-skill-attack',ID);

        }

    });

    socket.on('ranger_skill', function (data) {
        let sendingdata = {
            sendingid : data.name,
            point: {
                worldX:data.x,
                worldY:data.y,
            }
        };
        if(players[data.name].ranger_cool_time === false) { //
            console.log("레인저스킬 사용");
            players[data.name].ranger_cool_time = true;

            let count = 3;

            io.to(connected_players[data.name].socketid).emit('ranger_skill_using', count, 'start');
            setInterval(function () {

                count--;
                io.to(connected_players[data.name].socketid).emit('ranger_skill_using', count, 'during');

                if (count <= 0) {
                    clearInterval(this);
                    if (players[data.name] !== undefined) {
                        players[data.name].ranger_cool_time = false;
                        io.to(connected_players[data.name].socketid).emit('ranger_skill_using', count, 'stop');
                        io.to(connected_players[data.name].socketid).emit('broadcast_game_info', '얼음화살 쿨타임이 회복되었습니다.', 'skill');

                    }
                }
            }, 1000);

            io.emit('update-players', players);
            io.to(connected_players[data.name].socketid).emit('broadcast_game_info', '얼음화살 스킬을 사용하였습니다.', 'skill');
            io.emit('update-ranger-skill-attack',sendingdata);

        }

        //



    });

    socket.on('healer_skill', function (ID) {
        if(players[ID].healer_skill === false){ // 쿨타임 끝나기 전에 못쓰도록
            console.log("힐러스킬 사용");
            players[ID].healer_skill = true;
            players[ID].heal_effect = true;

            let healvalue = Math.floor(Math.random()*100 + 50);
            players[ID].heal_value = healvalue;

            if(players[ID].MaxHP.val === players[ID].HP.val){
                    io.to(connected_players[ID].socketid).emit('broadcast_game_info', '이미 체력이 가득차있습니다.', 'skill');

            }else {
                players[ID].HP.val += healvalue;
                if(players[ID].MaxHP.val < players[ID].HP.val){
                    players[ID].HP.val = players[ID].MaxHP.val;
                    io.to(connected_players[ID].socketid).emit('broadcast_game_info', '자가치유 스킬을 사용합니다. 체력이 완전히 회복되었습니다.', 'skill');

                }else {
                    io.to(connected_players[ID].socketid).emit('broadcast_game_info', '자가치유 스킬을 사용합니다. 체력이 '+ healvalue+ '만큼 회복합니다!!', 'skill');
                }

                io.emit('update-players', players);
                io.to(connected_players[ID].socketid).emit('update_hp', players[ID].HP.val, players[ID].MaxHP.val);
            }


            let buff_time = 1;
            let count = buff_time;


            io.to(connected_players[ID].socketid).emit('healer_skill_using', count, 'start', healvalue);
            setInterval(function () {

                count--;
                io.to(connected_players[ID].socketid).emit('healer_skill_using', count, 'during');

                if( count <= 0){
                    clearInterval(this);
                    io.to(connected_players[ID].socketid).emit('healer_skill_using', count, 'stop');
                }

            }, 1000);

            setTimeout(dd(), 1000);
            function dd() {
                return function(){
                    console.log("이펙트끝");
                    players[ID].heal_effect = false;
                    io.emit('update-players', players);
                };
            }


            setTimeout(descreaseDM(), buff_time*1000);
            function descreaseDM() {
                return function(){
                    console.log("힐러스킬끝");
                    players[ID].healer_skill = false;

                    io.emit('update-players', players);
                    io.to(connected_players[ID].socketid).emit('broadcast_game_info', '자가치유스킬의 쿨타임이 회복되었습니다.', 'skill');

                };
            }
            io.emit('update-players', players);
        }
    });

    socket.on('ninja_skill', function (ID) {
        if(players[ID].ninja_skill === false && players[ID].ninja_cool_time === false){ // 쿨타임 끝나기 전에 못쓰도록
            console.log("닌자스킬 사용");
            players[ID].ninja_skill = true;
            players[ID].ninja_cool_time = true;

            let buff_time = 5;
            let count = 10;

            io.to(connected_players[ID].socketid).emit('warrior_skill_using', count, 'start');
            setInterval(function () {

                count--;
                io.to(connected_players[ID].socketid).emit('warrior_skill_using', count, 'during');

                if( count <= 0){
                    clearInterval(this);
                    if( players[ID] !== undefined) {
                        players[ID].ninja_cool_time = false;
                        io.to(connected_players[ID].socketid).emit('warrior_skill_using', count, 'stop');
                        io.to(connected_players[ID].socketid).emit('broadcast_game_info', '투명스킬의 쿨타임이 회복되었습니다.', 'skill');

                    }
                }

            }, 1000);

            setTimeout(descreaseDM(), buff_time*1000);
            function descreaseDM() {
                return function(){
                    if( players[ID] === undefined){
                        return
                    }else {
                        console.log("닌자 투명 해제");
                        players[ID].ninja_skill = false;

                        io.emit('update-players', players);
                        io.to(connected_players[ID].socketid).emit('broadcast_game_info', '투명스킬의 지속시간이 끝났습니다.', 'skill');
                    }


                };
            }
            io.emit('update-players', players);
            io.to(connected_players[ID].socketid).emit('broadcast_game_info', '투명스킬을 사용하였습니다.' + buff_time + '초 동안 남에게 보이지 않습니다.', 'skill');
        }
    });




    //=================================================================================================================

    //클라이언트의 연결이 끊겼을때 콜백
    socket.on('disconnect', state => {
        for(let id in connected_players){
            if(connected_players[id].socketid === socket.id){
                console.log("나간 플레이어 : ", id );
                delete players[id];
                io.emit('update-players', players);

            }
        }
        console.log(chat_list[socket.id], socket.id,  "와 연결이 끊겼습니다.");

        let tmp = chat_list[socket.id];
        delete chat_list[socket.id];
        io.emit('chat_number',chat_list, socket.id, tmp);
    });


    //채팅 관련
    console.log('user connected: ', socket.id);  //3-1
    //io.to(socket.id).emit('change_name',name);   //3-1
    io.emit('update_connection',chat_list, socket.id);
    io.emit('chat_number',chat_list, null);

    /*socket.on('disconnect', function(){ //3-2
        console.log('user disconnected: ', socket.id);
    });*/

    socket.on('send_message', function(name,text){ //3-3
        var msg ={ name: name, message: text};
        console.log(msg);
        io.emit('receive_message', msg);
    });

});


function TierGrade(point) {
/* 티어점수
    0 ~ 99 : 브론즈(0)        100 ~ 199 : 실버(1)        200 ~ 299 : 골드(2)        300 ~ 399 : 플래티넘(3)
    400 ~ 499 : 다이아몬드(4)   500 ~ 599 : 마스터(5)       600 ~ 699 : 그랜드마스터(6)  700~ : 신(7)
    */

    if(point >= 0 && point < 100){
        return 0;
    }else if( point >= 100 && point < 200){
        return 1;

    }else if (point >= 200 && point < 300){
        return 2;

    } else if (point >= 300 && point < 400){
        return 3;

    } else if (point >= 400 && point < 500){
        return 4;

    }else if (point >= 500 && point < 600){
        return 5;

    } else if(point >= 600 && point < 700){
        return 6;

    }else if(point >= 700) {
        return 7;

    }
}


/*
io.on('connection', socket => {
socket.on('logindata', data =>{
   console.log("login id : " + idd, socket.id);
   tmp = {id: idd, socket: socket.id};
   //connected_players[idd].socketid = socket.id;
    //console.log("저장된 데이터2\n", connected_players);
    io.to(socket.id).emit('loginID', tmp);
});

socket.on('new-game', () => {
    console.log('GameBasicData');
    let playerData = makePlayerData();

    let Data = { GameBasicData: GameBasicData, playerData: playerData };
    io.emit('generate-world', Data);
});

socket.on('new-player', state => {
    console.log('New player joined with state:\n', state , '\n-----------------------------------\n');
    //console.log('palyerbasicdata:\n', playerData);

    players[socket.id] = state;
    // Emit the update-players method in the client side
    io.emit('update-players', players);
    console.log("현재 접속중인 플레이어 수 : " + Object.keys(players).length);

});

socket.on('move-player', data => {
    //console.log('move player  : ' + socket.id);

    const { x, y, angle, playerName, speed, HP} = data;

    // If the player is invalid, return
    if (players[socket.id] === undefined) {
        return
    }

    // Update the player's data if he moved
    players[socket.id].x = x;
    players[socket.id].y = y;
    players[socket.id].angle = angle;
    players[socket.id].playerName = {
        name: playerName.name,
        x: playerName.x,
        y: playerName.y
    };
    players[socket.id].speed = {
        value: speed.value,
        x: speed.x,
        y: speed.y
    };

    players[socket.id].HP = {
        val: HP.val
    };
    // Send the data back to the client
    io.emit('update-players', players);
});

socket.on('attack-player', function(data){
    //console.log("receive attack!!");

    // console.log('attack to '+socket.id+', '+data.x + " / " + data.y);

    if(players[socket.id] === undefined){
        //console.log('attack to return');
        return
    }
    let sendingdata = {
        sendingid : socket.id,
        point: {
            worldX:data.x,
            worldY:data.y,
        }
    };

    io.emit('update-attack',sendingdata);
});

/!*socket.on('damage-player-to-otherplayer', function (data) {
    console.log("receive damage : " + data.attackID + " / " + data.damagedID);

    if(players[socket.id] === undefined){
        return
    }

    if(data.attackID === socket.id){
        players[data.damagedID].HP.val -= 10;
        console.log("check : " + players[data.damagedID].playerName.name + " / " + players[data.damagedID].HP.val);

        io.emit('update-players', players);
    }

});*!/

socket.on('damage-player-to', function (data) {
    console.log("공격 : [ " + data.attackID + " ] -> [ " + data.damagedID + " ]");

    if(players[socket.id] === undefined){
        return
    }

    /!*if(data.attackID === socket.id){
        players[data.damagedID].HP.val -= 10;

        //캐릭터가 죽었을때 클라이언트에게 콜백보내는 부분
        if(players[data.damagedID].HP.val <= 0){
            //console.log("died player : " + data.damagedID + " / " + Object.keys(players).length);
            let tempdata = players[data.damagedID];
            io.emit('died-player', tempdata);
            delete players[data.damagedID];

        }
        //console.log("check2 : " + players[data.damagedID].playerName.name + " / " + players[data.damagedID].HP.val);

        io.emit('update-players', players);
    }*!/

    let attackID = connected_players[data.attackID].socketid;
    let damagedID = connected_players[data.damagedID].socketid;

    if(attackID === socket.id){
        players[attackID].HP.val -= 10;

        //캐릭터가 죽었을때 클라이언트에게 콜백보내는 부분
        if(players[damagedID].HP.val <= 0){
            //console.log("died player : " + data.damagedID + " / " + Object.keys(players).length);
            let tempdata = players[damagedID];
            io.emit('died-player', tempdata);
            delete players[damagedID];

        }
        //console.log("check2 : " + players[data.damagedID].playerName.name + " / " + players[data.damagedID].HP.val);

        io.emit('update-players', players);
    }

});


    /!*socket.on('nowPosition',function(data){
        //console.log('click to '+data.posX+', '+data.posY);
        const index = getPlayerInfoIndexById(data.id);

        if(index > -1){
            PLAYER_INFOS[index].x = data.posX;
            PLAYER_INFOS[index].y = data.posY;
        }


        //io.emit('move',socket.player);
    });

*!/
    /!*socket.on('attack',function(data){
        //console.log('attack to '+data.swordx+', '+data.swordy);

        io.emit('attacker',socket.player);
    });*!/

socket.on('disconnect', state => {
    console.log("나간 플레이어 : ",players[socket.id] );
    delete players[socket.id];
    io.emit('update-players', players)

});

    /!*socket.on('disconnect',function(){
        io.emit('remove',socket.player.id);
        console.log("remove id " + socket.player.id);

        /!*const index = getPlayerInfoIndexById(socket.player.id);

        if(index > -1){
            PLAYER_INFOS.splice(index, 1);
        }*!/
        //delete PLAYER_INFOS[index];
    });*!/

    /!*socket.on('inputKeyboard',function(number){
        socket.player.keys = number;
        io.emit('movePlayer',socket.player);
    });

socket.on('mapData',function(){
    console.log('map received');
    io.emit('makedMapData', dungeon);
});


socket.on('test',function(){
    console.log('test received');
});*!/

});
*/
