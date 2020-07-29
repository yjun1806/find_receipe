package com.example.capstone;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

import java.util.List;

public class ResultDataForm {
        @SerializedName("classes")
        @Expose
        private List<String> classes = null;
        @SerializedName("predict_number")
        @Expose
        private List<String> predictNumber = null;

        public List<String> getClasses() {
            return classes;
        }

        public void setClasses(List<String> classes) {
            this.classes = classes;
        }

        public List<String> getPredictNumber() {
            return predictNumber;
        }

        public void setPredictNumber(List<String> predictNumber) {
            this.predictNumber = predictNumber;
        }

    }
