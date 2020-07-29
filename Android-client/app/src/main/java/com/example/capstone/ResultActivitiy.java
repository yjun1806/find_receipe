package com.example.capstone;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;

import com.google.gson.Gson;

import java.util.HashMap;

public class ResultActivitiy extends AppCompatActivity {
    private ImageView result_img1, result_img2, result_img3;
    private TextView result_name1, result_name2, result_name3;
    private TextView result_per1, result_per2, result_per3;
    private CardView result_card1, result_card2, result_card3;
    private final HashMap<String, String> category_name = new HashMap<>();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result_activitiy);

        init(); // 각종 초기화 함수

        Intent intent = getIntent(); // 이전 액티비티에서 전달받은 데이터를 불러온다.

        /**
         * JSON 형태이기 때문에 GSON을 이용해 자바 변수로 활용할 수 있도록 바꿔준다.
         * */
        Gson gson = new Gson();
        final ResultDataForm resultDataForm = gson.fromJson(intent.getStringExtra("result"), ResultDataForm.class);

        // 카테고리 이름을 뷰에 셋팅해준다.
        result_name1.setText(category_name.get(resultDataForm.getClasses().get(0)));
        result_name2.setText(category_name.get(resultDataForm.getClasses().get(1)));
        result_name3.setText(category_name.get(resultDataForm.getClasses().get(2)));


        /**
         * 서버로부터 넘겨받은 카테고리의 이름이 "000.재료명"으로 되어있는데, 숫자부분을 없애고 재료명만 분리해서 가져오고
         * 가져온 재료명으로 drawable 리소스에서 이름에 맞는
         * 재료명.jpg로 되어있는 이미지들을 이미지뷰에 셋팅해주는 부분이다.
         * */
        String[] re = resultDataForm.getClasses().get(0).split("\\.");
        String[] re2 = resultDataForm.getClasses().get(1).split("\\.");
        String[] re3 = resultDataForm.getClasses().get(2).split("\\.");
        result_img1.setImageResource(this.getResources().getIdentifier(re[1], "drawable", this.getPackageName()));
        result_img2.setImageResource(this.getResources().getIdentifier(re2[1], "drawable", this.getPackageName()));
        result_img3.setImageResource(this.getResources().getIdentifier(re3[1], "drawable", this.getPackageName()));


        /**
         * 예측치 %를 가져와서 텍스트뷰에 넣어주는 부분이다.
         * */
        float per1 = Float.parseFloat(resultDataForm.getPredictNumber().get(0));
        float per2 = Float.parseFloat(resultDataForm.getPredictNumber().get(1));
        float per3 = Float.parseFloat(resultDataForm.getPredictNumber().get(2));
        String t1 = String.valueOf(Math.round((per1*10000)/100.0));
        String t2 = String.valueOf(Math.round((per2*10000)/100.0));
        String t3 = String.valueOf(Math.round((per3*10000)/100.0));

        result_per1.setText(t1 + "%");
        result_per2.setText(t2 + "%");
        result_per3.setText(t3 + "%");


        /**
         * 만약 0% 확률이 나온다면 보여주는게 의미가 없으므로 카드뷰를 GONE 처리해준다.
         * */
        if(Float.parseFloat(t1) <= 0.0) result_card1.setVisibility(View.GONE);
        if(Float.parseFloat(t2) <= 0.0) result_card2.setVisibility(View.GONE);
        if(Float.parseFloat(t3) <= 0.0) result_card3.setVisibility(View.GONE);



        /**
         * 분석된 결과나 나온 카드뷰를 터치하면 레시피 크롤링 액티비티로 연결되도록 만드는 부분
         * */
        result_card1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent1 = new Intent(v.getContext(), RecipeListActivity.class);
                intent1.putExtra("category", category_name.get(resultDataForm.getClasses().get(0)));
                startActivity(intent1);

            }
        });

        result_card2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent1 = new Intent(v.getContext(), RecipeListActivity.class);
                intent1.putExtra("category", category_name.get(resultDataForm.getClasses().get(1)));
                startActivity(intent1);
            }
        });

        result_card3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent1 = new Intent(v.getContext(), RecipeListActivity.class);
                intent1.putExtra("category", category_name.get(resultDataForm.getClasses().get(2)));
                startActivity(intent1);
            }
        });


    }

    private void init(){
        result_img1 = findViewById(R.id.result_class_img_1);
        result_img2 = findViewById(R.id.result_class_img_2);
        result_img3 = findViewById(R.id.result_class_img_3);
        result_name1 = findViewById(R.id.result_title_1);
        result_name2 = findViewById(R.id.result_title_2);
        result_name3 = findViewById(R.id.result_title_3);
        result_per1 = findViewById(R.id.result_per_1);
        result_per2 = findViewById(R.id.result_per_2);
        result_per3 = findViewById(R.id.result_per_3);
        result_card1 = findViewById(R.id.result_1);
        result_card2 = findViewById(R.id.result_2);
        result_card3 = findViewById(R.id.result_3);

        category_name.put("001.anchovy", "멸치");
        category_name.put("002.apple", "사과");
        category_name.put("003.beef", "소고기");
        category_name.put("004.carrot", "당근");
        category_name.put("005.chicken", "닭");
        category_name.put("006.chili", "고추");
        category_name.put("007.chives", "부추");
        category_name.put("008.cockle", "꼬막");
        category_name.put("009.cucumber", "오이");
        category_name.put("010.daikon", "무");
        category_name.put("011.egg", "계란");
        category_name.put("012.eggplant", "가지");
        category_name.put("013.garlic", "마늘");
        category_name.put("014.greenonion", "대파");
        category_name.put("015.greenpumpkin", "애호박");
        category_name.put("016.kelp", "다시마");
        category_name.put("017.kingoystermushroom", "새송이버섯");
        category_name.put("018.laver", "김");
        category_name.put("019.onion", "양파");
        category_name.put("020.oyster", "굴");
        category_name.put("021.paprika", "파프리카");
        category_name.put("022.perillaleaf", "깻잎");
        category_name.put("023.pork", "돼지고기");
        category_name.put("024.potato", "감자");
        category_name.put("025.quailegg", "메추리알");
        category_name.put("026.shiitake", "표고버섯");
        category_name.put("027.shrimp", "새우");
        category_name.put("028.smallgreenonion", "쪽파");
        category_name.put("029.spinach", "시금치");
        category_name.put("030.squid", "오징어");
    }

    public void onClick_back_again_btn(View view) {
        Intent intent = new Intent(this, MainActivity.class);
        startActivity(intent);
        finish(); // 다시시작을 할 것이기 떄문에 기존의 액티비티는 살아있으면 안된다.
    }
}
