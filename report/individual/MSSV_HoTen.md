# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:*  2A202603491*
- **Họ và tên:* Nguyễn Vũ Hoàng *
- **SEED** (in ở ô đầu notebook): 4436

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | có |

## 2. Kết quả chính

Mô hình	F1 test	F1 SUB 40	Train (s)	Tin/s (cả lô)	p95 (ms/tin)	USD/1k tin
1. BoW + Logistic Regression	0.892	0.788	6.45	29000.9	0.088	0.00
2. BoW + MLP	0.935	0.947	8.55	45508.2	0.360	0.00
3a. Encoder (MiniLM) + kNN	0.879	0.947	94.84	25.6	66.807	0.00
3b. Encoder (MiniLM) + LogReg	0.929	0.974	95.28	29.9	40.529	0.00
3c. Encoder (MiniLM) + MLP head	0.923	0.947	95.85	29.6	39.617	0.00
4a–4c. LLM zero-shot/few-shot	Chưa có kết quả hoàn chỉnh	—	—	—	—	—


BoW + MLP có F1 test toàn bộ cao nhất (0.935), p95 rất thấp (0.360 ms) và tốc độ xử lý cả lô cao. Encoder + LogReg có F1 cao nhất trên SUB (0.974), nhưng chạy chậm hơn và p95 gần ngưỡng 50 ms. Thử nghiệm LLM gặp lỗi giới hạn số lượt API (429), nên tôi không ghi số F1 hoặc thời gian không có trong kết quả cuối.
## 3. Dự đoán và thực tế

Dự đoán 1 — ml_f1: Tôi dự đoán Logistic Regression đạt F1 khoảng 0.8. Kết quả thực tế trên toàn bộ tập test là 0.892 (precision 1.000, recall 0.805). Như vậy kết quả cao hơn dự đoán khoảng 0.092. Có thể do các từ đặc trưng trong bộ SMS đủ giúp mô hình tuyến tính phân loại khá tốt, dù vẫn bỏ sót một phần spam.

Dự đoán 2 — tradeoff: Tôi nghĩ BoW + MLP có thể nằm ngoài đường Pareto vì chậm hơn BoW + LogReg. Biểu đồ thực tế lại cho thấy BoW + MLP nằm trên Pareto, cùng BoW + LogReg và Encoder + LogReg. Tôi đã đánh giá thấp mức cải thiện F1 của MLP so với mức tăng độ trễ.

Dự đoán 3 — hard_eval: Tôi đoán F1 trên 50 case khó chỉ giảm khoảng 5–15 điểm phần trăm và BoW + LogReg giảm nhiều nhất. Thực tế BoW + MLP giảm mạnh nhất, từ F1 SUB 0.947 xuống 0.450 (giảm 49.7 điểm phần trăm); BoW + LogReg giảm từ 0.788 xuống 0.333 (giảm 45.5 điểm). Tôi nhận ra kết quả test thường có thể lạc quan hơn rất nhiều khi so với dữ liệu đánh lừa có chủ đích.

## 4. Phân tích 50 case khó

Mô hình	F1 trên SUB	F1 trên 50 case khó	Giảm (điểm phần trăm)
BoW + LogReg	0.788	0.333	45.5
BoW + MLP	0.947	0.450	49.7
Embedding + kNN	0.947	0.737	21.0
Embedding + LogReg	0.974	0.596	37.8
Embedding + MLP head	0.947	0.638	30.9

Mô hình giảm F1 nhiều nhất là BoW + MLP. Với nhóm né từ khóa, mô hình này đúng 0%; nhóm spam giọng tự nhiên cũng chỉ đúng 10%. Còn Embedding + kNN có F1 cao nhất trên 50 case khó (0.737), mặc dù không đáp ứng p95 dưới 50 ms trong kết quả đo.

Hai ví dụ sai cụ thể:
- Case #14 — spam giọng tự nhiên: Tin nhắn mở đầu như đang chia sẻ việc làm tại nhà, hứa thu nhập khoảng £300/ngày và không cần kinh nghiệm. Các mô hình dễ nhầm thành tin bình thường vì cách viết thân thiện, ít dùng các từ khóa spam quen thuộc.
- Case #41 — tin giao dịch hợp lệ (ham): Tin OTP đăng nhập kèm lời nhắc không chia sẻ mã và hướng dẫn liên hệ ngân hàng. Các mô hình có thể nhận nhầm thành spam vì chứa mã số và các cụm từ liên quan bảo mật/ngân hàng.
Tổng cộng có 8/50 case mà cả năm mô hình đều sai. Điều này cho thấy cần bổ sung dữ liệu tình huống khó và đánh giá riêng khả năng chặn nhầm tin giao dịch hợp lệ.
## 5. Thử thách cá nhân

Chưa làm được ạ ( không phải do lười đâu ạ em không làm được thật :( )
## 6. Phương án cho sản phẩm thật

Nhà mạng cần lọc 10 triệu SMS/ngày (trung bình khoảng 116 tin/giây), yêu cầu p95 < 50 ms/tin. Theo phép đo của tôi, bốn phương án đạt ngưỡng p95 là BoW + LogReg (0.088 ms), BoW + MLP (0.360 ms), Encoder + LogReg (40.529 ms) và Encoder + MLP head (39.617 ms). Encoder + kNN không đạt vì p95 = 66.807 ms.

Nếu ưu tiên triển khai nhanh và ổn định, tôi chọn BoW + MLP làm lớp lọc chính vì F1 test toàn bộ cao nhất (0.935) và throughput cả lô khoảng 45,508 tin/giây trong môi trường đo. Điểm yếu là F1 chỉ còn 0.450 trên bộ 50 case khó. Vì vậy tôi đề xuất bổ sung Encoder + LogReg cho những tin có xác suất phân loại không chắc chắn, thay vì sử dụng cho toàn bộ tin. Việc kết hợp cần triển khai có hàng đợi hoặc kiểm tra thêm để không làm vượt SLA; thực tế cũng phải đo lại khi có tải đồng thời.

Nếu gọi LLM cho toàn bộ tin nhắn và giả sử chi phí khoảng 0.019 USD/1.000 tin như điểm LLM trong biểu đồ Pareto, 10 triệu tin/ngày trong 30 ngày sẽ tốn khoảng 5,700 USD/tháng chỉ riêng API. Đây là ước tính tham khảo từ biểu đồ, không phải chi phí đã được xác nhận đầy đủ trong lab1_report.json. LLM cũng có độ trễ cao, nên không phù hợp làm bộ lọc đồng bộ duy nhất.
## 7. AI log và điều đã học

Phần	Tôi hỏi AI gì	AI trả lời/viết gì	Tôi sửa / kiểm chứng gì
Phần 1 — ML	Hỏi giải thích, code mẫu vectorize, sigmoid, gradient descent và các chỉ số	AI đưa ví dụ và các đoạn code tham khảo	Chạy ô kiểm tra 1.1–1.4 và các thí nghiệm 1.A/1.B; kiểm tra kết quả trên dữ liệu riêng
Phần 2 — DL	Hỏi 5 bước train PyTorch và lỗi indentation	AI giải thích zero_grad, forward, loss, backward, update và chỉ lỗi thụt lề	Sửa indentation và chạy kiểm tra TODO 2.1–2.2
Phần 3 — Embedding	Hỏi công thức cosine, top-k và kNN	AI giải thích và cung cấp ví dụ code	Chạy các ô kiểm tra 3.1–3.3 và so sánh kết quả mô hình
Phần 4 — LLM	Hỏi cách lấy API key, viết prompt và xử lý lỗi 429 / phản hồi None	AI đưa code tham khảo, giải thích quota và kiểm tra phản hồi rỗng	Mock test 4.1+4.4 đạt; thử một lời gọi Gemini trả ham trong 0.92 giây; các thí nghiệm API lớn chưa hoàn thành do giới hạn
Phần 5 — Pareto	Hỏi mô hình nào sẽ trên/ngoài Pareto	AI dự đoán BoW + MLP có thể nằm ngoài	Đối chiếu biểu đồ thật, thấy BoW + MLP nằm trên Pareto và sửa nhận xét
Phần 5 — Stress test	Hỏi mô hình nào giảm F1 mạnh nhất	AI dự đoán BoW + LogReg có thể giảm nhiều nhất	Đối chiếu bảng 50 case, phát hiện BoW + MLP mới giảm nhiều nhất (49.7 điểm phần trăm)


Điều tôi học được: Trước đây tôi hay nhìn accuracy và nghĩ điểm cao là mô hình tốt, nhưng dữ liệu lệch lớp khiến accuracy có thể gây hiểu nhầm. Tôi cũng hiểu vì sao cần đánh giá cả F1, p95, throughput và các case khó: mô hình có kết quả cao trên test thường vẫn có thể sai rất nhiều khi cách viết tin nhắn thay đổi. Quan trọng hơn, AI hỗ trợ viết code vẫn cần được kiểm tra và người làm bài phải hiểu chính xác từng bước.
