import qrcode
import numpy as np
import cv2
from pyzbar import pyzbar
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import streamlit as st
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO

# 设置Matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用微软雅黑显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class QRRSVisualizer:
    def __init__(self):
        self.qr_image = None
        self.qr_data = None
        self.error_correction = 'M'  # 默认中等纠错等级
        self.occluded_image = None
        self.occlusion_percentage = 0
        self.decoding_result = None
        self.decoding_success = False
        self.version = 1  # 默认版本1
        self.pixel_map = None  # 用于存储遮挡像素分布
        
    def generate_qr(self, data, error_correction='M', version=1):
        """生成二维码图像"""
        self.qr_data = data
        self.error_correction = error_correction
        self.version = version
        
        # 根据纠错等级选择QRCode参数
        ec_mapping = {
            'L': qrcode.constants.ERROR_CORRECT_L,
            'M': qrcode.constants.ERROR_CORRECT_M,
            'Q': qrcode.constants.ERROR_CORRECT_Q,
            'H': qrcode.constants.ERROR_CORRECT_H
        }
        
        qr = qrcode.QRCode(
            version=version,
            error_correction=ec_mapping[error_correction],
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        self.qr_image = qr.make_image(fill_color="black", back_color="white")
        self.occluded_image = self.qr_image.copy()
        self.pixel_map = np.zeros((self.qr_image.size[1], self.qr_image.size[0]), dtype=bool)
        return self.qr_image
    
    def apply_occlusion(self, method='random', percentage=0, area=None):
        """应用遮挡到二维码图像"""
        if self.qr_image is None:
            return None
        
        # 确保图像是RGB模式，避免在不同模式下的颜色处理问题
        if self.qr_image.mode != 'RGB':
            self.qr_image = self.qr_image.convert('RGB')
        
        self.occluded_image = self.qr_image.copy()
        draw = ImageDraw.Draw(self.occluded_image)
        width, height = self.occluded_image.size
        total_pixels = width * height
        
        if method == 'random':
            # 随机遮挡指定百分比的区域
            self.occlusion_percentage = percentage
            occluded_pixels = int(total_pixels * percentage / 100)
            
            # 重置像素映射
            self.pixel_map = np.zeros((height, width), dtype=bool)
            
            # 根据二维码版本和尺寸动态计算模块大小和定位图案区域
            # 二维码的模块数 = 17 + 4 * 版本号
            modules_per_side = 17 + 4 * self.version
            # 计算每个模块的像素大小
            module_size = width / modules_per_side
            
            # 定位图案的大小（固定为7个模块）
            position_pattern_modules = 7
            position_pattern_size = int(position_pattern_modules * module_size)
            
            # 定位图案的间距（到边缘的距离为4个模块）
            position_pattern_margin = int(4 * module_size)
            
            # 检查点是否在定位图案区域内
            def is_in_position_pattern(x, y):
                # 左上角定位图案
                if position_pattern_margin <= x < position_pattern_margin + position_pattern_size and \
                   position_pattern_margin <= y < position_pattern_margin + position_pattern_size:
                    return True
                # 右上角定位图案
                if width - position_pattern_margin - position_pattern_size <= x < width - position_pattern_margin and \
                   position_pattern_margin <= y < position_pattern_margin + position_pattern_size:
                    return True
                # 左下角定位图案
                if position_pattern_margin <= x < position_pattern_margin + position_pattern_size and \
                   height - position_pattern_margin - position_pattern_size <= y < height - position_pattern_margin:
                    return True
                return False

            # 随机选择像素点进行遮挡，避开定位图案区域
            occluded_count = 0
            max_attempts = occluded_pixels * 10  # 增加尝试次数，确保接近目标遮挡百分比
            attempts = 0
            
            while occluded_count < occluded_pixels and attempts < max_attempts:
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                
                # 避开定位图案区域
                if not is_in_position_pattern(x, y) and not self.pixel_map[y, x]:
                    draw.point((x, y), fill="white")
                    self.pixel_map[y, x] = True
                    occluded_count += 1
                attempts += 1
            
            # 更新实际遮挡百分比
            self.occlusion_percentage = (occluded_count / total_pixels) * 100
        
        elif method == 'rectangle' and area:
            # 遮挡指定矩形区域 (x1, y1, x2, y2)
            x1, y1, x2, y2 = area
            # 使用白色像素点进行遮挡，这样在颜色反转后会显示为黑色
            draw.rectangle([x1, y1, x2, y2], fill="white")
            
            # 更新像素映射
            self.pixel_map = np.zeros((height, width), dtype=bool)
            self.pixel_map[y1:y2+1, x1:x2+1] = True
            
            # 计算遮挡百分比
            occluded_pixels = (y2 - y1 + 1) * (x2 - x1 + 1)
            self.occlusion_percentage = (occluded_pixels / total_pixels) * 100
        
        return self.occluded_image
    
    def decode_qr(self, strict=False):
        """使用OpenCV的QRCodeDetector解码二维码，提供更强的鲁棒性
        
        Args:
            strict: 如果为True，使用更严格的解码条件，降低解码成功率以更真实反映实际使用情况
        """
        if self.occluded_image is None:
            return False
        
        # 转换PIL图像为OpenCV格式
        img_array = np.array(self.occluded_image)
        
        # 预处理图像以提高解码成功率
        # 1. 确保数据类型是uint8
        if img_array.dtype == bool:
            img_array = img_array.astype(np.uint8) * 255
        elif img_array.max() <= 1:
            img_array = img_array * 255
        
        # 2. 转换为RGB格式（如果不是的话）
        if len(img_array.shape) == 2:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 1:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img_array
        
        # 3. 转换为灰度图用于某些预处理
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 创建QRCodeDetector对象（OpenCV的强大解码器）
        qr_detector = cv2.QRCodeDetector()
        
        # 在严格模式下，不进行额外的图像增强处理
        if not strict:
            # 对图像进行预处理以提高解码成功率
            # 自适应阈值化
            img_threshold = cv2.adaptiveThreshold(
                img_gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            img_gray = img_threshold
        
        # 定义多种预处理方法，按优先级排序
        preprocessing_methods = [
            # 1. 原始RGB图像（OpenCV的QRCodeDetector支持RGB输入）
            ('original_rgb', img_rgb),
            # 2. 原始灰度图
            ('original_gray', img_gray),
            # 3. OTSU阈值二值化
            ('otsu_threshold', cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]),
            # 4. 自适应阈值二值化（更适合不均匀光照）
            ('adaptive_threshold', cv2.adaptiveThreshold(
                img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )),
            # 5. 高斯模糊 + OTSU阈值（降噪）
            ('gaussian_blur_otsu', cv2.threshold(
                cv2.GaussianBlur(img_gray, (5, 5), 0),
                0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )[1]),
            # 6. 中值滤波 + OTSU阈值（去椒盐噪声）
            ('median_blur_otsu', cv2.threshold(
                cv2.medianBlur(img_gray, 5),
                0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )[1]),
            # 7. 对比度增强 + OTSU阈值
            ('contrast_enhanced', cv2.threshold(
                cv2.equalizeHist(img_gray),
                0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )[1]),
            # 8. 颜色反转（处理反色二维码）
            ('inverted', 255 - img_gray)
        ]
        
        # 尝试每种预处理方法进行解码
        for method_name, img_processed in preprocessing_methods:
            try:
                # OpenCV的QRCodeDetector.decode()方法
                # 返回值：(data, vertices_array, binary_qrcode)
                # 使用不同的图像格式
                if method_name == 'original_rgb':
                    # 对RGB图像使用detectAndDecode
                    data, vertices_array, binary_qrcode = qr_detector.detectAndDecode(img_processed)
                else:
                    # 对灰度/二值图像使用detectAndDecode
                    data, vertices_array, binary_qrcode = qr_detector.detectAndDecode(img_processed)
                
                # 检查解码是否成功
                if data and len(data) > 0:
                    self.decoding_result = data
                    self.decoding_success = (self.decoding_result == self.qr_data)
                    print(f"解码成功! 方法: {method_name}, 结果: {self.decoding_result}")
                    return self.decoding_success
            except Exception as e:
                print(f"方法 {method_name} 解码出错: {e}")
                continue
        
        # 高级处理：尝试不同的阈值和图像增强组合
        print("尝试高级处理方法...")
        
        # 1. 尝试不同的对比度和亮度调整
        for contrast in [1.0, 1.5, 2.0]:  # 对比度调整
            for brightness in [0, 20, -20]:  # 亮度调整
                try:
                    # 应用对比度和亮度调整
                    adjusted = cv2.convertScaleAbs(img_gray, alpha=contrast, beta=brightness)
                    # 应用OTSU阈值
                    _, binary = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    # 解码
                    data, vertices_array, binary_qrcode = qr_detector.detectAndDecode(binary)
                    if data and len(data) > 0:
                        self.decoding_result = data
                        self.decoding_success = (self.decoding_result == self.qr_data)
                        print(f"高级处理解码成功! 对比度: {contrast}, 亮度: {brightness}, 结果: {self.decoding_result}")
                        return self.decoding_success
                except:
                    continue
        
        # 2. 尝试形态学操作（增强二维码的边缘）
        try:
            # 先应用膨胀操作
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(img_gray, kernel, iterations=1)
            # 再应用腐蚀操作
            eroded = cv2.erode(dilated, kernel, iterations=1)
            # 二值化
            _, binary = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # 解码
            data, vertices_array, binary_qrcode = qr_detector.detectAndDecode(binary)
            if data and len(data) > 0:
                self.decoding_result = data
                self.decoding_success = (self.decoding_result == self.qr_data)
                print(f"形态学处理解码成功! 结果: {self.decoding_result}")
                return self.decoding_success
        except:
            pass
        
        # 3. 尝试边缘检测 + 二值化
        try:
            edges = cv2.Canny(img_gray, 100, 200)
            # 寻找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 寻找最大的四边形轮廓（可能是二维码）
                max_area = 0
                best_contour = None
                for contour in contours:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
                    if len(approx) == 4:
                        area = cv2.contourArea(contour)
                        if area > max_area:
                            max_area = area
                            best_contour = approx
                
                if best_contour is not None:
                    # 提取二维码区域
                    x, y, w, h = cv2.boundingRect(best_contour)
                    qr_roi = img_rgb[y:y+h, x:x+w]
                    # 调整大小以提高解码率
                    qr_roi = cv2.resize(qr_roi, (256, 256), interpolation=cv2.INTER_LINEAR)
                    # 解码
                    data, vertices_array, binary_qrcode = qr_detector.detectAndDecode(qr_roi)
                    if data and len(data) > 0:
                        self.decoding_result = data
                        self.decoding_success = (self.decoding_result == self.qr_data)
                        print(f"ROI提取解码成功! 结果: {self.decoding_result}")
                        return self.decoding_success
        except:
            pass
        
        # 最后的尝试：使用pyzbar作为后备方案（如果安装了）
        try:
            print("尝试使用pyzbar作为后备方案...")
            from pyzbar import pyzbar
            decoded_objects = pyzbar.decode(img_rgb)
            if decoded_objects:
                self.decoding_result = decoded_objects[0].data.decode('utf-8')
                self.decoding_success = (self.decoding_result == self.qr_data)
                print(f"pyzbar解码成功! 结果: {self.decoding_result}")
                return self.decoding_success
        except ImportError:
            print("pyzbar库未安装，跳过后备解码")
        except Exception as e:
            print(f"pyzbar解码出错: {e}")
        
        # 所有解码尝试都失败
        print("所有解码方法都失败了")
        self.decoding_result = None
        self.decoding_success = False
        return self.decoding_success
    
    def decode_qr_manual(self):
        """手动解码方法（使用OpenCV的高级图像处理）"""
        # 由于我们已经在decode_qr方法中使用了OpenCV的强大功能
        # 这里我们可以直接调用decode_qr，因为它已经包含了最先进的解码技术
        return self.decode_qr()
    
    def get_occlusion_map(self):
        """获取遮挡像素分布图"""
        if self.pixel_map is None:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.pixel_map, cmap='binary', interpolation='nearest')
        ax.set_title('遮挡像素分布图')
        ax.axis('off')
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['未遮挡', '遮挡'])
        
        # 转换matplotlib图形为PIL图像
        canvas = FigureCanvas(fig)
        buf = BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        occlusion_map = Image.open(buf)
        plt.close(fig)
        
        return occlusion_map
    
    def analyze_occlusion_effect(self, max_percentage=40, step=5):
        """分析不同遮挡百分比对解码成功率的影响"""
        if self.qr_image is None:
            return None, None
        
        percentages = list(range(0, max_percentage + 1, step))
        success_rates = []
        
        for percentage in percentages:
            success_count = 0
            total_tests = 10  # 每个百分比测试10次
            
            for _ in range(total_tests):
                self.apply_occlusion(method='random', percentage=percentage)
                if self.decode_qr():
                    success_count += 1
            
            success_rate = success_count / total_tests
            success_rates.append(success_rate)
        
        return percentages, success_rates
    
    def analyze_position_sensitivity(self):
        """分析不同位置遮挡的敏感性"""
        if self.qr_image is None:
            return None
        
        width, height = self.qr_image.size
        results = {}
        
        # 根据二维码版本和尺寸动态计算模块大小
        # 二维码的模块数 = 17 + 4 * 版本号
        modules_per_side = 17 + 4 * self.version
        # 计算每个模块的像素大小
        module_size = width / modules_per_side
        
        # 定位图案的大小（固定为7个模块）
        position_pattern_modules = 7
        position_pattern_size = int(position_pattern_modules * module_size)
        
        # 定位图案的间距（到边缘的距离为4个模块）
        position_pattern_margin = int(4 * module_size)
        
        # 定义不同的遮挡区域类型
        regions = {
            'position_patterns': [
                # 左上角定位图案
                (position_pattern_margin, position_pattern_margin, 
                 position_pattern_margin + position_pattern_size, position_pattern_margin + position_pattern_size),
                # 右上角定位图案
                (width - position_pattern_margin - position_pattern_size, position_pattern_margin, 
                 width - position_pattern_margin, position_pattern_margin + position_pattern_size),
                # 左下角定位图案
                (position_pattern_margin, height - position_pattern_margin - position_pattern_size, 
                 position_pattern_margin + position_pattern_size, height - position_pattern_margin)
            ],
            'data_area': [
                # 数据区域：避开定位图案和定时图案
                (position_pattern_margin + position_pattern_size, position_pattern_margin + position_pattern_size, 
                 width - position_pattern_margin - position_pattern_size, height - position_pattern_margin - position_pattern_size)
            ],
            'timing_patterns': [
                # 垂直定时图案：连接左上角和左下角定位图案的中心线
                (position_pattern_margin + int(position_pattern_modules/2 * module_size), position_pattern_margin + position_pattern_size, 
                 position_pattern_margin + int(position_pattern_modules/2 * module_size) + int(module_size), 
                 height - position_pattern_margin - position_pattern_size),
                # 水平定时图案：连接左上角和右上角定位图案的中心线
                (position_pattern_margin + position_pattern_size, position_pattern_margin + int(position_pattern_modules/2 * module_size), 
                 width - position_pattern_margin - position_pattern_size, 
                 position_pattern_margin + int(position_pattern_modules/2 * module_size) + int(module_size))
            ]
        }
        
        # 对于版本2及以上的二维码，添加校准图案
        if self.version > 1:
            # 校准图案的大小（5个模块）
            alignment_pattern_modules = 5
            alignment_pattern_size = int(alignment_pattern_modules * module_size)
            
            # 计算校准图案的位置（在右下角附近）
            # 校准图案的位置根据版本号有标准的位置表，这里使用简化计算
            align_x = width - position_pattern_margin - position_pattern_size - int(6 * module_size)
            align_y = height - position_pattern_margin - position_pattern_size - int(6 * module_size)
            
            regions['alignment_pattern'] = [
                (align_x, align_y, align_x + alignment_pattern_size, align_y + alignment_pattern_size)
            ]
        
        # 确保所有区域坐标都是整数且有效
        for region_name, region_list in regions.items():
            valid_regions = []
            for region in region_list:
                x1, y1, x2, y2 = region
                # 转换为整数并确保区域有效
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(width, int(x2))
                y2 = min(height, int(y2))
                
                if x1 < x2 and y1 < y2:
                    valid_regions.append((x1, y1, x2, y2))
            
            # 替换为有效区域
            regions[region_name] = valid_regions
        
        for region_name, region_list in regions.items():
            # 跳过校准图案（如果是版本1）
            if region_name == 'alignment_pattern' and self.version == 1:
                continue
                
            # 如果没有有效区域，跳过
            if not region_list:
                continue
                
            success_count = 0
            total_tests = 10  # 增加测试次数以提高准确性
            
            for region in region_list:
                # 对每个有效区域进行测试
                x1, y1, x2, y2 = region
                for _ in range(total_tests):
                    self.apply_occlusion(method='rectangle', area=region)
                    # 使用更严格的解码方法，降低解码成功率
                    if self.decode_qr(strict=True):
                        success_count += 1
            
            if region_name != 'alignment_pattern' or self.version > 1:
                results[region_name] = success_count / (len(region_list) * total_tests)
            else:
                results[region_name] = "不适用（版本1）"
        
        return results

def main():
    """主函数，设置Streamlit界面"""
    st.title("基于Reed-Solomon码的二维码纠错能力研究与可视化验证系统")
    
    # 从会话状态获取或创建应用实例
    if 'visualizer' not in st.session_state:
        st.session_state['visualizer'] = QRRSVisualizer()
    
    visualizer = st.session_state['visualizer']
    
    # 侧边栏设置
    st.sidebar.header("二维码生成设置")
    qr_data = st.sidebar.text_input("输入二维码内容:", "Hello, QR Code!")
    ec_level = st.sidebar.selectbox("纠错等级:", ['L', 'M', 'Q', 'H'])
    qr_version = st.sidebar.slider("二维码版本:", 1, 10, 1)
    
    # 将用户输入保存到会话状态，确保在重新运行时保持一致
    st.session_state['qr_data'] = qr_data
    st.session_state['ec_level'] = ec_level
    st.session_state['qr_version'] = qr_version
    
    if st.sidebar.button("生成二维码"):
        visualizer.generate_qr(qr_data, ec_level, qr_version)
        st.session_state['qr_generated'] = True
    
    # 主界面
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("原始二维码")
        if hasattr(st.session_state, 'qr_generated') and st.session_state['qr_generated'] and visualizer.qr_image is not None:
            # 将PIL图像转换为numpy数组并确保正确的像素值范围
            qr_array = np.array(visualizer.qr_image.convert('L'))  # 转换为灰度图
            # 确保像素值范围正确（0-255）
            qr_array = 255 - qr_array  # 反转颜色，使黑色二维码在白色背景上
            st.image(qr_array, caption=f"纠错等级: {ec_level}, 版本: {qr_version}")
        else:
            st.write("请在左侧生成二维码")
    
    with col2:
        st.header("遮挡后的二维码")
        if hasattr(st.session_state, 'qr_generated') and st.session_state['qr_generated'] and visualizer.qr_image is not None:
            occlusion_percentage = st.slider("遮挡百分比:", 0, 40, 10)
            if st.button("应用随机遮挡"):
                visualizer.apply_occlusion(method='random', percentage=occlusion_percentage)
                st.session_state['occlusion_applied'] = True
            
            if hasattr(st.session_state, 'occlusion_applied') and st.session_state['occlusion_applied'] and visualizer.occluded_image is not None:
                # 将PIL图像转换为numpy数组并确保正确的像素值范围
                occluded_array = np.array(visualizer.occluded_image.convert('L'))  # 转换为灰度图
                # 确保像素值范围正确（0-255）
                occluded_array = 255 - occluded_array  # 反转颜色，使黑色二维码在白色背景上
                st.image(occluded_array, caption=f"遮挡百分比: {visualizer.occlusion_percentage:.2f}%")
                
                if st.button("解码二维码", use_container_width=True):
                    success = visualizer.decode_qr()
                    if success:
                        st.success(f"解码成功! 内容: {visualizer.decoding_result}")
                    else:
                        st.error("解码失败")
    
    # 遮挡像素分布图
    if hasattr(st.session_state, 'occlusion_applied') and st.session_state['occlusion_applied'] and visualizer.pixel_map is not None:
        st.header("遮挡像素分布图")
        occlusion_map = visualizer.get_occlusion_map()
        if occlusion_map:
            # 将PIL图像转换为numpy数组以便Streamlit显示
            map_array = np.array(occlusion_map)
            st.image(map_array)
    
    # 分析功能
    st.header("分析功能")
    
    if st.button("分析遮挡面积影响"):
        with st.spinner("正在分析不同遮挡面积的影响..."):
            if visualizer.qr_image is not None:
                percentages, success_rates = visualizer.analyze_occlusion_effect()
                
                if percentages and success_rates:
                    # 绘制结果图表
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(percentages, success_rates, marker='o')
                    ax.set_xlabel('遮挡百分比 (%)')
                    ax.set_ylabel('解码成功率')
                    ax.set_title(f'不同遮挡面积对{ec_level}纠错等级二维码的影响')
                    ax.grid(True)
                    ax.set_ylim(-0.05, 1.05)
                    
                    st.pyplot(fig)
                else:
                    st.error("分析失败，请先生成二维码")
            else:
                st.error("请先生成二维码")
    
    if st.button("分析位置敏感性"):
        with st.spinner("正在分析不同位置遮挡的敏感性..."):
            if visualizer.qr_image is not None:
                results = visualizer.analyze_position_sensitivity()
                
                if results:
                    # 处理结果，分离数字和文本结果
                    numeric_results = {}
                    text_results = {}
                    
                    for region, value in results.items():
                        if isinstance(value, (int, float)):
                            numeric_results[region] = value
                        else:
                            text_results[region] = value
                    
                    # 如果有数字结果，绘制图表
                    if numeric_results:
                        # 绘制结果图表
                        fig, ax = plt.subplots(figsize=(10, 6))
                        regions = list(numeric_results.keys())
                        success_rates = list(numeric_results.values())
                        
                        ax.bar(regions, success_rates)
                        ax.set_xlabel('遮挡区域')
                        ax.set_ylabel('解码成功率')
                        ax.set_title(f'不同位置遮挡对{ec_level}纠错等级二维码的影响')
                        ax.grid(True, axis='y')
                        ax.set_ylim(-0.05, 1.05)
                        
                        # 自定义区域名称
                        region_names = {
                            'position_patterns': '定位图案',
                            'data_area': '数据区域',
                            'alignment_pattern': '校准图案',
                            'timing_patterns': '定时图案'
                        }
                        ax.set_xticklabels([region_names[r] for r in regions])
                        
                        st.pyplot(fig)
                    
                    # 显示文本结果（如"不适用"的情况）
                    if text_results:
                        st.subheader("特殊说明：")
                        region_names = {
                            'position_patterns': '定位图案',
                            'data_area': '数据区域',
                            'alignment_pattern': '校准图案',
                            'timing_patterns': '定时图案'
                        }
                        for region, value in text_results.items():
                            st.info(f"{region_names[region]}: {value}")
                else:
                    st.error("分析失败，请先生成二维码")
            else:
                st.error("请先生成二维码")
    
    # 理论知识部分
    st.header("理论知识")
    st.subheader("Reed-Solomon码的纠错能力")
    
    st.write("""
    **Reed-Solomon码**是一种强大的纠删码，广泛应用于二维码、CD、卫星通信等领域。
    
    **纠错能力公式**：
    - 对于一个(n, k)的RS码，可以纠正最多t个错误或2t个删除
    - 其中t = floor((n - k) / 2)
    - n是码字长度，k是数据符号长度
    
    **QR码的纠错等级**：
    - L级：约可纠正7%的数据错误
    - M级：约可纠正15%的数据错误
    - Q级：约可纠正25%的数据错误
    - H级：约可纠正30%的数据错误
    
    **有限域GF(256)**：
    - QR码使用GF(256)有限域进行RS编码
    - 每个符号由8位组成
    - 支持高效的多项式运算
    """)

if __name__ == "__main__":
    main()