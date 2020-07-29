const STATS_OCCUPATION = { //각 클래스별 속성들 설정하는 부분
    /*
    * '공 격 력 : ★★★☆☆\n체     력 : ★★★★★\n이동속도 : ★★☆☆☆\n공격속도 : ★★★☆☆', //워리어

      '공 격 력 : ★★★★★\n체     력 : ★★☆☆☆\n이동속도 : ★★★☆☆\n공격속도 : ★★☆☆☆', //메이지

      '공 격 력 : ★★★☆☆\n체     력 : ★★★☆☆\n이동속도 : ★★★★☆\n공격속도 : ★★★★☆', //레인저

      '공 격 력 : ★☆☆☆☆\n체     력 : ★★★★☆\n이동속도 : ★★★☆☆\n공격속도 : ★★★☆☆', // 힐러

      '공 격 력 : ★★☆☆☆\n체     력 : ★★★☆☆\n이동속도 : ★★★★★\n공격속도 : ★★★★★', // 닌자
    * */

    warrior_m: {
        ATTACK_POWER: 40,
        HP: 1000,
        SPEED: 200,
        ATTACK_SPEED: 300,
        ATTACK_RANGE: 300
    },
    warrior_f: {
        ATTACK_POWER: 40,
        HP: 800,
        SPEED: 300,
        ATTACK_SPEED: 300,
        ATTACK_RANGE: 300
    },
    mage_m: {
        ATTACK_POWER: 100,
        HP: 300,
        SPEED: 250,
        ATTACK_SPEED: 400,
        ATTACK_RANGE: 700
    },
    mage_f: {
        ATTACK_POWER: 100,
        HP: 200,
        SPEED: 350,
        ATTACK_SPEED: 400,
        ATTACK_RANGE: 700
    },
    ranger_m: {
        ATTACK_POWER: 60,
        HP: 350,
        SPEED: 500,
        ATTACK_SPEED: 700,
        ATTACK_RANGE: 800
    },
    ranger_f: {
        ATTACK_POWER: 40,
        HP: 300,
        SPEED: 500,
        ATTACK_SPEED: 600,
        ATTACK_RANGE: 1000
    },
    healer_m: {
        ATTACK_POWER: 10,
        HP: 700,
        SPEED: 250,
        ATTACK_SPEED: 200,
        ATTACK_RANGE: 500
    },
    healer_f: {
        ATTACK_POWER: 10,
        HP: 500,
        SPEED: 350,
        ATTACK_SPEED: 200,
        ATTACK_RANGE: 500
    },
    ninja_m: {
        ATTACK_POWER: 30,
        HP: 300,
        SPEED: 400,
        ATTACK_SPEED: 150,
        ATTACK_RANGE: 300
    },
    ninja_f: {
        ATTACK_POWER: 20,
        HP: 200,
        SPEED: 500,
        ATTACK_SPEED: 70,
        ATTACK_RANGE: 300
    }
};

module.exports = STATS_OCCUPATION;