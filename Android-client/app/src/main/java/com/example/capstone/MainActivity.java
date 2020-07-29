package com.example.capstone;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import com.google.gson.Gson;
import com.gun0912.tedpermission.PermissionListener;
import com.gun0912.tedpermission.TedPermission;
import com.orhanobut.logger.Logger;
import com.theartofdev.edmodo.cropper.CropImage;
import com.theartofdev.edmodo.cropper.CropImageView;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;

public class MainActivity extends AppCompatActivity {
    private static final int FROM_CAMERA = 0;
    private static final int FROM_ALBUM = 1;
    private ImageView preview;

    private String pick_img_url ="";
    private File tempFile;
    private boolean img_selected = false;
    private boolean isCamera;
    public boolean flag = true;
    private String result_data;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tedPermission(); // 권한 요청 라이브러리 호출


    }

    private void tedPermission(){
        //TedPermission 라이브러리 -> 카메라 권한 획득
        PermissionListener permissionlistener = new PermissionListener() {
            @Override
            public void onPermissionGranted() {
                //Toast.makeText(MainActivity.this, "권한 허가", Toast.LENGTH_SHORT).show();
            }
            @Override
            public void onPermissionDenied(List<String> deniedPermissions) {
                //Toast.makeText(MainActivity.this, "권한 거절\n" + deniedPermissions.toString(), Toast.LENGTH_SHORT).show();
            }
        };

        TedPermission.with(this) // 권한요청
                .setPermissionListener(permissionlistener)
                .setDeniedMessage("If you reject permission,you can not use this service\n\nPlease turn on permissions at [Setting] > [Permission]")
                .setPermissions(Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.INTERNET) // 요청할 권한 적기
                .check();
    }

    public void onClick_camera_btn(View view) {
        getImgfromCamera();
    }

    public void onClick_gallery_btn(View view) {
        getImgfromGallery();
    }


    @RequiresApi(api = Build.VERSION_CODES.Q)
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            switch (requestCode) {
                case FROM_ALBUM: { // 앨범에서 이미지 선택 후의 처리
                    Uri photoUri = data.getData();
                    Logger.d("PICK_FROM_ALBUM photoUri : " + photoUri);
                    cropImage(photoUri);

                    break;
                }
                case FROM_CAMERA: { // 카메라에서 이미지를 찍은 후의 처리
                    Uri photoUri = Uri.fromFile(tempFile);
                    Logger.d("takePhoto photoUri : " + photoUri);
                    cropImage(photoUri);

                    break;
                }
                case CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE: { // 이미지 크롭 후의 처리
                    CropImage.ActivityResult result = CropImage.getActivityResult(data);
                    setImage(result.getUri()); // 이미지를 미리보기 이미지뷰에 세팅시켜준다.
                    Logger.d("getUri : " + result.getUri());
                }
            }
        } if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE) {
            CropImage.ActivityResult result = CropImage.getActivityResult(data);
            Exception error = result.getError();
        }else {
            Logger.d("resultCode Error!!");
        }
    }



    private String getRealPathFromURI(Uri contentURI) {
        /*실제 이미지 경로를 가져오는 함수
        * 일반적으로 file 경로를 가져오면 content:// 로 되어있는데 이런 경우 서버로 전송할때 에러가 발생하므로
        * 실제 파일 경로를 가져오도록 한다.
        * */
        String filePath;
        Cursor cursor = this.getContentResolver().query(contentURI, null, null, null, null);
        if (cursor == null) {
            filePath = contentURI.getPath();
        } else {
            cursor.moveToFirst();
            int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
            filePath = cursor.getString(idx);
            cursor.close();
        }
        return filePath;
    }

    private void uploadToServer(String filePath) {
        /*서버에 파일 전송하는 함수*/

        Retrofit retrofit = NetworkClient.getRetrofitClient(this);
        UploadAPIs uploadAPIs = retrofit.create(UploadAPIs.class);
        //Create a file object using file path
        File file = new File(filePath);
        // Create a request body with file and image media type
        RequestBody fileReqBody = RequestBody.create(MediaType.parse("image/*"), file);
        // Create MultipartBody.Part using file request-body,file name and part name
        MultipartBody.Part part = MultipartBody.Part.createFormData("upload", file.getName(), fileReqBody);
        //Create request body with text description and text media type
        RequestBody description = RequestBody.create(MediaType.parse("text/plain"), "image-type");
        //
        Call call = uploadAPIs.uploadImage(part, description);
        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                try {
                    Logger.d("onResponse : " + response.body());
                    result_data = response.body().string();
                   /* Gson gson = new Gson();
                    ResultDataForm resultDataForm = gson.fromJson(response.body().string(), ResultDataForm.class);
                    Logger.d(resultDataForm.getClasses().get(0));*/
                    flag = false;
                } catch (IOException e) {
                    e.printStackTrace();
                }

            }
            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                Logger.d("onFailure : " + t);
            }
        });
    }

    private void getImgfromGallery() {
        isCamera = false;
        Intent albumintent = new Intent(Intent.ACTION_PICK); // 카메라 캡쳐를 실행할 인텐트 생성
        albumintent.setDataAndType(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(albumintent, FROM_ALBUM); // 액티비티 실행 후 결과를 받는 메소드
    }

    private void getImgfromCamera() {
        isCamera = true;

        // 카메라 사진 찍는 버튼을 눌렀을때 동작 부분
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE); // 카메라 캡쳐를 실행할 인텐트 생성

        try {
            tempFile = createImageFile();
        } catch (IOException e) {
            Toast.makeText(this, "이미지 처리 오류! 다시 시도해주세요.", Toast.LENGTH_SHORT).show();
            finish();
            e.printStackTrace();
        }
        if (tempFile != null) {

            /**
             *  안드로이드 OS 누가 버전 이후부터는 file:// URI 의 노출을 금지로 FileUriExposedException 발생
             *  Uri 를 FileProvider 도 감싸 주어야 합니다.
             *
             *  참고 자료 http://programmar.tistory.com/4 , http://programmar.tistory.com/5
             */

            // 카메라로 찍는 경우 capstone이라는 폴더에 저장된다.
            // 갤러리에서 선택하는 경우는 저장되지 않음.
            // 크롭된 사진은 저장되지 않는다. 임시캐시폴더에 저장되고 나중에 지워질 수 있다.
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                Uri photoUri = FileProvider.getUriForFile(this,"com.example.capstone", tempFile);
                cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
                startActivityForResult(cameraIntent, FROM_CAMERA);

            } else {
                Uri photoUri = Uri.fromFile(tempFile);
                Logger.d("takePhoto photoUri : " + photoUri);
                cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
                startActivityForResult(cameraIntent, FROM_CAMERA);

            }
        }
    }

    private File createImageFile() throws IOException {

        // 이미지 파일 이름 ( capstone {시간}_ )
        String timeStamp = new SimpleDateFormat("HHmmss").format(new Date());
        String imageFileName = "capstone" + timeStamp + "_";

        // 이미지가 저장될 파일 이름 ( capstone )
        File storageDir = new File(Environment.getExternalStorageDirectory() + "/capstone/");
        if (!storageDir.exists()) storageDir.mkdirs();

        // 빈 파일 생성
        File image = File.createTempFile(imageFileName, ".jpg", storageDir);
        Logger.d("createImageFile : " + image.getAbsolutePath());

        return image;
    }

    private void cropImage(Uri photoUri) {
        // 이미지 크롭 라이브러리 사용 함수
        CropImage.activity(photoUri)
                .setGuidelines(CropImageView.Guidelines.ON)
                .start(this);
    }

    private void setImage(Uri img_uri) {
        TextView textView = findViewById(R.id.main_step2);
        textView.setVisibility(View.VISIBLE);
        preview = findViewById(R.id.main_selected_img_view);
        pick_img_url = getRealPathFromURI(img_uri);
        preview.setImageURI(Uri.parse(pick_img_url));
        img_selected = true;
    }


    public void onClick_Sendimg_btn(View view) {
        if(img_selected) {
            uploadToServer(pick_img_url);
            CheckTypesTask task = new CheckTypesTask();
            task.execute();


        }else {
            Toast.makeText(this, "먼저 사진을 선택해주세요." ,Toast.LENGTH_SHORT).show();
        }
    }

    private class CheckTypesTask extends AsyncTask<Void, Void, Void> {

        ProgressDialog asyncDialog = new ProgressDialog(
                MainActivity.this);


        /**
         * onPreExecute() : 작업시작, ProgressDialog 객체를 생성하고 시작합니다.
         * doInBackground() : 진행중, ProgressDialog 의 진행 정도를 표현해 줍니다.
         * doPostExecute() : 종료, ProgressDialog 종료 기능을 구현합니다.
         * */

        @Override
        protected void onPreExecute() {
            asyncDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
            asyncDialog.setMessage("분석중입니다..");

            // show dialog
            asyncDialog.show();
            asyncDialog.setCanceledOnTouchOutside(false);
            super.onPreExecute();
        }

        @Override
        protected Void doInBackground(Void... arg0) {
            try {
                /**
                 * 서버에 이미지를 보내고 분석된 결과를 돌려받는 동안 동작하는 부분이다.
                 * 분석중입니다라는 메세지와 로딩다이얼로그가 뜨도록해준다.
                 * flag로 인해 데이터를 받게되면 반복문을 벗어나도록 되어있다.*/
                while(flag){
                    Thread.sleep(300);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return null;
        }

        @Override
        protected void onPostExecute(Void result) {
            asyncDialog.dismiss();
            super.onPostExecute(result);

            /** 서버에서 분석한 데이터가 도착하면 실행되는 부분이다.
             * 새로운 액티비티를 실행시켜 분석 결과를 표시해준다.*/
            Intent intent = new Intent(MainActivity.this, ResultActivitiy.class);
            intent.putExtra("result", result_data);
            startActivity(intent);
            finish(); // 분석결과를 보는 화면으로 넘어갔기 떄문에 이 액티비티는 더이상 쓸모가 없다.
        }
    }


}
