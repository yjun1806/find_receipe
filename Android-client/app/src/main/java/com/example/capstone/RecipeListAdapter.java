package com.example.capstone;

import android.content.Context;
import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.cardview.widget.CardView;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;

import java.util.ArrayList;

public class RecipeListAdapter extends RecyclerView.Adapter<RecipeListAdapter.ViewHolder> {
    private ArrayList<RecipeData> crawlingdata = null;

    RecipeListAdapter(ArrayList<RecipeData> data){
        this.crawlingdata = data;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        Context context = parent.getContext();
        LayoutInflater inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);

        View view = inflater.inflate(R.layout.layout_recipe_list, parent, false);
        RecipeListAdapter.ViewHolder vh = new RecipeListAdapter.ViewHolder(view);

        return vh;
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, final int position) {
        String url = crawlingdata.get(position).getUrl();
        String title = crawlingdata.get(position).getTitle();
        String aut = crawlingdata.get(position).getAuthor();

        Glide.with(holder.itemView.getContext()).load(url).override(500).into(holder.img);
        holder.title.setText(title);
        holder.author.setText(aut);
        holder.card.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(v.getContext(), RecipeViewActivity.class);
                intent.putExtra("url", crawlingdata.get(position).getLink());
                v.getContext().startActivity(intent);
            }
        });
    }

    @Override
    public int getItemCount() {
        return crawlingdata.size();
    }

    public class ViewHolder extends RecyclerView.ViewHolder {
        ImageView img;
        TextView title, author;
        CardView card;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            img = itemView.findViewById(R.id.recycler_img_view);
            title = itemView.findViewById(R.id.recycler_text_title);
            author = itemView.findViewById(R.id.recycler_text_author);
            card = itemView.findViewById(R.id.recycler_card);
        }
    }
}
