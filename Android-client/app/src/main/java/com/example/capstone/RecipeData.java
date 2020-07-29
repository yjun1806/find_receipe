package com.example.capstone;

public class RecipeData {
    private String img_url;
    private String title;
    private String author;
    private String link;

    RecipeData(String img, String ti, String a, String link){
        this.img_url = img;
        this.title = ti;
        this.author = a;
        this.link = link;
    }


    public String getUrl(){
        return this.img_url;
    }

    public String getTitle(){
        return this.title;
    }

    public String getAuthor(){
        return this.author;
    }

    public String getLink(){
        return this.link;
    }
}
