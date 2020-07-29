package com.example.capstone;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.app.ProgressDialog;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.widget.TextView;

import com.orhanobut.logger.Logger;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.IOException;
import java.util.ArrayList;


public class RecipeListActivity extends AppCompatActivity {

    private String page_url = "http://www.10000recipe.com/recipe/list.html?q=";
    private String category_url = "";
    private ArrayList<RecipeData> crawlingData = null;
    RecyclerView recyclerView;
    RecipeListAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recipe_list);

        Intent intent = getIntent();
        String category = intent.getStringExtra("category");
        category_url = page_url + category;
        Logger.d("URL : " + category_url);

        crawlingData = new ArrayList<>();
        adapter = new RecipeListAdapter(crawlingData);
        recyclerView = findViewById(R.id.recipe_recyclerview);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(adapter);




        TextView name = findViewById(R.id.receipe_category_name);
        name.setText(category + "요리 레시피");

        JsoupAsynctask jsoupAsyncTask = new JsoupAsynctask();
        jsoupAsyncTask.execute();
    }


    private class JsoupAsynctask extends AsyncTask<Void, Void, Void>{
        ProgressDialog asyncDialog = new ProgressDialog(RecipeListActivity.this);

        @Override
        protected void onPreExecute() {
            super.onPreExecute();

            asyncDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
            asyncDialog.setMessage("잠시만 기다려주세요..");
            // show dialog
            asyncDialog.show();
            asyncDialog.setCanceledOnTouchOutside(false);
        }

        @Override
        protected Void doInBackground(Void... voids) {
            try {
                Document doc = Jsoup.connect(category_url).get();
                //Logger.i(String.valueOf(doc));


                /** 크롤링 데이터
                 * │ <div class="col-xs-3">
                 * │  <a class="thumbnail" href="/recipe/6886842">
                 *     <span class="thumbnail_over">
                 *         <img src="http://recipe1.ezmember.co.kr/img/thumb_over.png">
         *             </span>
                 *     <img src="레시피 썸네일 이미지 주소" style="width:275px; height:275px;">
                 * │   <div class="caption">
                 * │    <h4 class="ellipsis_title2">레시피 제목 </h4>
                 * │    <p>레시피 작성자</p>
                 * │   </div> </a>
                 * │  <div style="position:absolute;top:365px;width:100%;text-align:right;right:20px;">
                 * │   <span style="color:#74b243;font-size:10px;" class="glyphicon glyphicon-certificate"></span>
                 * │  </div>
                 * │ </div>
                * */
                Elements elements = doc.select("div[class=rcp_m_list2]").select("div[class=row]").select("div[class=col-xs-3]");
                //Logger.i(String.valueOf(elements));
                //Logger.d("Size : " + elements.size());
                for(Element e: elements){
                    Logger.i(String.valueOf(e));
                    if(!e.select("ins").hasClass("adsbygoogle")) { // 광고부분을 크롤링하는 경우 제외처리
                        String href = e.select("a[class=thumbnail]").attr("href");
                        Elements es = e.getElementsByTag("img");
                        String imgsrc = "";
                        if (es.size() != 0) {
                            imgsrc = es.get(1).attr("src");
                        }
                        String title = e.select("h4").text();
                        String by = e.select("p").text();

                        if(title != null || title != "") {
                            crawlingData.add(new RecipeData(imgsrc, title, by, " http://www.10000recipe.com" + href));
                        }
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            Logger.d("Size : " + crawlingData.size());
            adapter.notifyDataSetChanged();

            asyncDialog.dismiss();

        }
    }
}
