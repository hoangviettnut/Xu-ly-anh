#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp> 

using namespace std;
using namespace cv;

// ==========================================================
// PHẦN 1: CÀI ĐẶT THỦ CÔNG THUẬT TOÁN CANNY
// ==========================================================

Mat lamMinGaussian(const Mat& anhGoc) {
    Mat anhKq = anhGoc.clone();
    double kernel[5][5] = { {2,4,5,4,2}, {4,9,12,9,4}, {5,12,15,12,5}, {4,9,12,9,4}, {2,4,5,4,2} };
    double tongSo = 159.0;
    for (int i = 2; i < anhGoc.rows - 2; i++) {
        for (int j = 2; j < anhGoc.cols - 2; j++) {
            double giaTri = 0;
            for (int ki = -2; ki <= 2; ki++) {
                for (int kj = -2; kj <= 2; kj++) {
                    giaTri += anhGoc.at<uchar>(i + ki, j + kj) * kernel[ki + 2][kj + 2];
                }
            }
            anhKq.at<uchar>(i, j) = (uchar)(giaTri / tongSo);
        }
    }
    return anhKq;
}

void tinhGradient(const Mat& anhLamMin, Mat& doLon, Mat& huong) {
    doLon = Mat::zeros(anhLamMin.size(), CV_32F);
    huong = Mat::zeros(anhLamMin.size(), CV_32F);
    int Gx[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,1} };
    int Gy[3][3] = { {-1,-2,-1}, {0,0,0}, {1,2,1} };
    for (int i = 1; i < anhLamMin.rows - 1; i++) {
        for (int j = 1; j < anhLamMin.cols - 1; j++) {
            float tongX = 0, tongY = 0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int pixel = anhLamMin.at<uchar>(i + ki, j + kj);
                    tongX += pixel * Gx[ki + 1][kj + 1];
                    tongY += pixel * Gy[ki + 1][kj + 1];
                }
            }
            doLon.at<float>(i, j) = sqrt(tongX * tongX + tongY * tongY);
            huong.at<float>(i, j) = atan2(tongY, tongX) * 180.0 / CV_PI;
        }
    }
}

Mat trietTieuPhiCucDai(const Mat& doLon, const Mat& huong) {
    Mat kq = Mat::zeros(doLon.size(), CV_8U);
    for (int i = 1; i < doLon.rows - 1; i++) {
        for (int j = 1; j < doLon.cols - 1; j++) {
            float goc = huong.at<float>(i, j);
            if (goc < 0) goc += 180;
            float q = 255, r = 255;
            if ((goc >= 0 && goc < 22.5) || (goc >= 157.5 && goc <= 180)) { q = doLon.at<float>(i, j + 1); r = doLon.at<float>(i, j - 1); }
            else if (goc >= 22.5 && goc < 67.5) { q = doLon.at<float>(i + 1, j - 1); r = doLon.at<float>(i - 1, j + 1); }
            else if (goc >= 67.5 && goc < 112.5) { q = doLon.at<float>(i + 1, j); r = doLon.at<float>(i - 1, j); }
            else if (goc >= 112.5 && goc < 157.5) { q = doLon.at<float>(i - 1, j - 1); r = doLon.at<float>(i + 1, j + 1); }

            if (doLon.at<float>(i, j) >= q && doLon.at<float>(i, j) >= r) kq.at<uchar>(i, j) = (uchar)doLon.at<float>(i, j);
            else kq.at<uchar>(i, j) = 0;
        }
    }
    return kq;
}

Mat locNguongVaTheoDoi(const Mat& anhManh, int nguongThap, int nguongCao) {
    Mat kq = Mat::zeros(anhManh.size(), CV_8U);
    uchar YEU = 50; uchar MANH = 255;
    for (int i = 0; i < anhManh.rows; i++) {
        for (int j = 0; j < anhManh.cols; j++) {
            int giaTri = anhManh.at<uchar>(i, j);
            if (giaTri >= nguongCao) kq.at<uchar>(i, j) = MANH;
            else if (giaTri >= nguongThap) kq.at<uchar>(i, j) = YEU;
        }
    }
    for (int i = 1; i < kq.rows - 1; i++) {
        for (int j = 1; j < kq.cols - 1; j++) {
            if (kq.at<uchar>(i, j) == YEU) {
                if (kq.at<uchar>(i + 1, j - 1) == MANH || kq.at<uchar>(i + 1, j) == MANH || kq.at<uchar>(i + 1, j + 1) == MANH ||
                    kq.at<uchar>(i, j - 1) == MANH || kq.at<uchar>(i, j + 1) == MANH || kq.at<uchar>(i - 1, j - 1) == MANH ||
                    kq.at<uchar>(i - 1, j) == MANH || kq.at<uchar>(i - 1, j + 1) == MANH) {
                    kq.at<uchar>(i, j) = MANH;
                }
                else kq.at<uchar>(i, j) = 0;
            }
        }
    }
    for (int i = 0; i < kq.rows; i++) {
        for (int j = 0; j < kq.cols; j++) {
            if (kq.at<uchar>(i, j) == YEU) kq.at<uchar>(i, j) = 0;
        }
    }
    return kq;
}

// ==========================================================
// PHẦN 2: CẤU TRÚC QUADTREE (MÃ TỨ PHÂN)
// ==========================================================

struct NodeCayTuPhan {
    int giaTri;
    NodeCayTuPhan* con[4];
    NodeCayTuPhan(int val) {
        giaTri = val;
        for (int i = 0; i < 4; i++) con[i] = nullptr;
    }
};

int kiemTraDongNhat(const Mat& anh, int x, int y, int rong, int cao) {
    if (rong <= 0 || cao <= 0) return -1;
    Mat vung = anh(Rect(x, y, rong, cao));
    int diemSang = countNonZero(vung);
    if (diemSang == 0) return 0;
    if (diemSang == rong * cao) return 255;
    return -1;
}

NodeCayTuPhan* xayDungCayTuPhan(const Mat& anh, int x, int y, int rong, int cao) {
    if (rong <= 0 || cao <= 0) return nullptr;
    int loaiNode = kiemTraDongNhat(anh, x, y, rong, cao);

    if (loaiNode != -1 || (rong <= 1 && cao <= 1)) {
        if (loaiNode == -1) loaiNode = anh.at<uchar>(y, x);
        return new NodeCayTuPhan(loaiNode);
    }

    NodeCayTuPhan* node = new NodeCayTuPhan(-1);
    int nuaRong = rong / 2, nuaCao = cao / 2;
    int rong2 = rong - nuaRong, cao2 = cao - nuaCao;

    node->con[0] = xayDungCayTuPhan(anh, x, y, nuaRong, nuaCao);
    node->con[1] = xayDungCayTuPhan(anh, x + nuaRong, y, rong2, nuaCao);
    node->con[2] = xayDungCayTuPhan(anh, x, y + nuaCao, nuaRong, cao2);
    node->con[3] = xayDungCayTuPhan(anh, x + nuaRong, y + nuaCao, rong2, cao2);
    return node;
}

void maHoaCay(NodeCayTuPhan* node, vector<bool>& chuoiBit) {
    if (!node) return;
    if (node->giaTri == -1) {
        chuoiBit.push_back(true);
        for (int i = 0; i < 4; i++) maHoaCay(node->con[i], chuoiBit);
    }
    else {
        chuoiBit.push_back(false);
        chuoiBit.push_back(node->giaTri == 255);
    }
}

// ==========================================================
// PHẦN 3: GIẢI MÃ TỪ BITSTREAM
// ==========================================================

NodeCayTuPhan* giaiMaCay(const vector<bool>& chuoiBit, int& viTri, int rong, int cao) {
    if (rong <= 0 || cao <= 0) return nullptr;
    if (viTri >= chuoiBit.size()) return nullptr;

    bool laNhanh = chuoiBit[viTri++];

    if (rong <= 1 && cao <= 1) {
        if (laNhanh) {
            if (viTri < chuoiBit.size()) viTri++;
            return new NodeCayTuPhan(0);
        }
        else {
            if (viTri >= chuoiBit.size()) return new NodeCayTuPhan(0);
            bool laTrang = chuoiBit[viTri++];
            return new NodeCayTuPhan(laTrang ? 255 : 0);
        }
    }

    if (laNhanh) {
        NodeCayTuPhan* node = new NodeCayTuPhan(-1);
        int nuaRong = rong / 2, nuaCao = cao / 2;
        int rong2 = rong - nuaRong, cao2 = cao - nuaCao;

        node->con[0] = giaiMaCay(chuoiBit, viTri, nuaRong, nuaCao);
        node->con[1] = giaiMaCay(chuoiBit, viTri, rong2, nuaCao);
        node->con[2] = giaiMaCay(chuoiBit, viTri, nuaRong, cao2);
        node->con[3] = giaiMaCay(chuoiBit, viTri, rong2, cao2);
        return node;
    }
    else {
        if (viTri >= chuoiBit.size()) return new NodeCayTuPhan(0);
        bool laTrang = chuoiBit[viTri++];
        return new NodeCayTuPhan(laTrang ? 255 : 0);
    }
}

void veLaiAnhTuCay(NodeCayTuPhan* node, Mat& anh, int x, int y, int rong, int cao) {
    if (!node || rong <= 0 || cao <= 0) return;
    if (node->giaTri != -1) {
        anh(Rect(x, y, rong, cao)).setTo(node->giaTri);
        return;
    }
    int nuaRong = rong / 2, nuaCao = cao / 2;
    int rong2 = rong - nuaRong, cao2 = cao - nuaCao;
    veLaiAnhTuCay(node->con[0], anh, x, y, nuaRong, nuaCao);
    veLaiAnhTuCay(node->con[1], anh, x + nuaRong, y, rong2, nuaCao);
    veLaiAnhTuCay(node->con[2], anh, x, y + nuaCao, nuaRong, cao2);
    veLaiAnhTuCay(node->con[3], anh, x + nuaRong, y + nuaCao, rong2, cao2);
}

void giaiPhongCay(NodeCayTuPhan* node) {
    if (!node) return;
    for (int i = 0; i < 4; i++) giaiPhongCay(node->con[i]);
    delete node;
}

// ==========================================================
// HÀM MAIN
// ==========================================================

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    // --- CẤU HÌNH ĐƯỜNG DẪN TỆP TIN ---
    string path_anhGoc = "D://Xu_ly_anh//baitaplon//anhgoc.jpg";
    string path_anhBien = "D://Xu_ly_anh//baitaplon//anh_bien.jpg";
    string path_fileNen = "D://Xu_ly_anh//baitaplon//anh_da_nen.bin";
    string path_anhGiaiMa = "D://Xu_ly_anh//baitaplon//anh_giai_ma.jpg";

    cout << "--- BAT DAU CHUONG TRINH ---" << endl;

    Mat anhGoc = imread(path_anhGoc, IMREAD_GRAYSCALE);
    if (anhGoc.empty()) {
        cout << "[Loi] Khong tim thay anh goc! Kiem tra lai duong dan." << endl;
        cin.get(); return -1;
    }

    // 1. TRÍCH XUẤT BIÊN BẰNG CANNY
    cout << "1. Dang trich xuat bien bang thuat toan Canny..." << endl;
    Mat anhLamMin = lamMinGaussian(anhGoc);
    Mat doLon, huong;
    tinhGradient(anhLamMin, doLon, huong);
    Mat anhManh = trietTieuPhiCucDai(doLon, huong);
    Mat anhBien = locNguongVaTheoDoi(anhManh, 60, 120);
    imwrite(path_anhBien, anhBien);

    // 2. MÃ HÓA QUADTREE VÀ GHI RA FILE BIN
    cout << "2. Dang nen anh bang Quadtree vao file .bin..." << endl;
    NodeCayTuPhan* gocCay = xayDungCayTuPhan(anhBien, 0, 0, anhBien.cols, anhBien.rows);
    vector<bool> chuoiBit;
    maHoaCay(gocCay, chuoiBit);

    ofstream file(path_fileNen, ios::binary);
    int w = anhBien.cols, h = anhBien.rows;
    file.write((char*)&w, sizeof(w));
    file.write((char*)&h, sizeof(h));

    unsigned char boDem = 0;
    int demBit = 0;
    for (bool bit : chuoiBit) {
        if (bit) boDem |= (1 << (7 - demBit));
        demBit++;
        if (demBit == 8) { file.write((char*)&boDem, 1); boDem = 0; demBit = 0; }
    }
    if (demBit > 0) file.write((char*)&boDem, 1);

    long sizeFileNen = file.tellp();
    file.close();

    // 3. ĐỌC VÀ GIẢI MÃ TỪ FILE BIN
    cout << "3. Dang doc va giai ma tu file .bin..." << endl;
    ifstream fileDoc(path_fileNen, ios::binary);

    int wDoc, hDoc;
    fileDoc.read((char*)&wDoc, sizeof(wDoc));
    fileDoc.read((char*)&hDoc, sizeof(hDoc));
    vector<bool> bitDocDuoc;

    unsigned char byteDoc;
    while (fileDoc.read((char*)&byteDoc, 1)) {
        for (int i = 7; i >= 0; i--) bitDocDuoc.push_back((byteDoc >> i) & 1);
    }
    fileDoc.close();

    int viTri = 0;
    NodeCayTuPhan* gocGiaiMa = giaiMaCay(bitDocDuoc, viTri, wDoc, hDoc);

    Mat anhPhucHoi = Mat::zeros(hDoc, wDoc, CV_8U);
    veLaiAnhTuCay(gocGiaiMa, anhPhucHoi, 0, 0, wDoc, hDoc);
    imwrite(path_anhGiaiMa, anhPhucHoi);

    // ==========================================================
    // TÍNH TOÁN CÁC CHỈ SỐ THỐNG KÊ
    // ==========================================================
    double sizeAnhGoc = w * h;
    double cr = sizeAnhGoc / (double)sizeFileNen;

    double mse = 0.0;
    for (int i = 0; i < hDoc; i++) {
        for (int j = 0; j < wDoc; j++) {
            double diff = (double)anhBien.at<uchar>(i, j) - (double)anhPhucHoi.at<uchar>(i, j);
            mse += diff * diff;
        }
    }
    mse = mse / (double)(wDoc * hDoc);

    double psnr = 0.0;
    if (mse == 0) psnr = INFINITY;
    else psnr = 10.0 * log10((255.0 * 255.0) / mse);

    cout << "\n==================================================" << endl;
    cout << "            THONG KE CHI SO THUC NGHIEM            " << endl;
    cout << "==================================================" << endl;
    cout << "- Kich thuoc anh bien goc   : " << sizeAnhGoc << " Bytes" << endl;
    cout << "- Kich thuoc file .bin      : " << sizeFileNen << " Bytes" << endl;
    cout << "- Ty le nen (CR)            : " << cr << " (Giam " << (1.0 - 1.0 / cr) * 100 << "%)" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "- Sai so toan phuong (MSE)  : " << mse << endl;
    if (mse == 0) {
        cout << "- Ty so tin hieu (PSNR)     : Vo cuc (Infinity) dB" << endl;
        cout << "  => KET LUAN: Nen Lossless (Khong mat mat) 100%" << endl;
    }
    else cout << "- Ty so tin hieu (PSNR)     : " << psnr << " dB" << endl;
    cout << "==================================================\n" << endl;

    giaiPhongCay(gocCay);
    giaiPhongCay(gocGiaiMa);

    cout << "=> XU LY HOAN TAT! Hien thi anh..." << endl;
    namedWindow("Anh Bien", WINDOW_NORMAL); resizeWindow("Anh Bien", 600, 600);
    namedWindow("Anh Giai Ma", WINDOW_NORMAL); resizeWindow("Anh Giai Ma", 600, 600);
    imshow("Anh Bien", anhBien);
    imshow("Anh Giai Ma", anhPhucHoi);
    waitKey(0);

    return 0;
}