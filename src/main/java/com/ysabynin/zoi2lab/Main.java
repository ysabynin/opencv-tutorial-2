package com.ysabynin.zoi2lab;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;

public class Main {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        String filePath = null;
        Integer numClusters = null;

        try {
            if (args.length == 2) {
                boolean isCorrect = true;
                filePath = args[0];
                File f = new File(filePath);
                if (!f.exists() || f.isDirectory())
                    isCorrect = false;

                numClusters = Integer.parseInt(args[1]);
                if (numClusters == null || numClusters < 0)
                    isCorrect = false;

                if (!isCorrect) {
                    throw new IllegalArgumentException();
                }
            } else throw new IllegalArgumentException();
        } catch (Exception e) {
            System.out.println("Since you passed wrong parameters program will finish execution!");
            return;
        }

        //Read Image
        Mat image = Imgcodecs.imread(filePath);
        Mat samples = image.reshape(1, image.cols() * image.rows());
        Mat samples32f = new Mat();
        samples.convertTo(samples32f, CvType.CV_32F, 1.0 / 255.0);

        //CREATE Criteria to stop algorithm and apply K-MEANS
        Mat labels = new Mat();
        TermCriteria criteria = new TermCriteria(TermCriteria.COUNT, 10, 1.0);
        Mat centers = new Mat();
        Core.kmeans(samples32f, numClusters, labels, criteria, 10, Core.KMEANS_PP_CENTERS, centers);

        //CONVERT TO PREVIOUS IMAGE
        centers.convertTo(centers, CvType.CV_8UC1, 255.0);
        centers.reshape(3);
        Mat dst = image.clone();
        int rows = 0;
        for (int y = 0; y < image.rows(); y++) {
            for (int x = 0; x < image.cols(); x++) {
                int label = (int) labels.get(rows, 0)[0];
                int r = (int) centers.get(label, 2)[0];
                int g = (int) centers.get(label, 1)[0];
                int b = (int) centers.get(label, 0)[0];
                dst.put(y, x, b, g, r);
                rows++;
            }
        }

        String filename = "result.bmp";
        Imgcodecs.imwrite(filename, dst);
    }
}
