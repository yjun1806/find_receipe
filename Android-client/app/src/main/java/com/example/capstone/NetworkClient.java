package com.example.capstone;

import android.content.Context;

import okhttp3.OkHttpClient;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class NetworkClient {
    /*Network Security Config 관련 주의 사항
    * https 가 아닌 http로 통신하는 경우 안드로이드 Pie 이상에서는 설정을 따로 해줘야 한다.
    * 접속하려는 사이트를 http여도 통신이 되도록 허용하는 부분을 추가해줘야 한다.
    * */
    private static final String BASE_URL = "http://kunde9999.iptime.org:15000/";
    private static Retrofit retrofit;
    public static Retrofit getRetrofitClient(Context context) {
        if (retrofit == null) {
            OkHttpClient okHttpClient = new OkHttpClient.Builder()
                    .build();
            retrofit = new Retrofit.Builder()
                    .baseUrl(BASE_URL)
                    .client(okHttpClient)
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();
        }
        return retrofit;
    }
}
