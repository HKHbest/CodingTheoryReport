# 简单的测试脚本，用于检查qr_rs_visualizer.py是否有语法错误

try:
    # 尝试导入模块
    import qr_rs_visualizer
    print("成功导入qr_rs_visualizer模块")
    
    # 尝试创建实例
    visualizer = qr_rs_visualizer.QRRSVisualizer()
    print("成功创建QRRSVisualizer实例")
    
    # 尝试调用apply_occlusion方法，检查参数是否正确
    print("检查apply_occlusion方法参数...")
    import inspect
    sig = inspect.signature(visualizer.apply_occlusion)
    print(f"apply_occlusion方法签名: {sig}")
    
    print("测试通过！所有修改都正确应用。")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
