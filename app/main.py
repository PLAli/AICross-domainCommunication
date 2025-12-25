"""
FastAPI主应用
提供REST API接口
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os

from app.ai_engine import translate_pm_to_dev, translate_dev_to_pm, get_translator

# 创建FastAPI应用
app = FastAPI(
    title="沟通翻译助手",
    description="产品经理和开发工程师之间的沟通翻译工具",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class TranslationRequest(BaseModel):
    """翻译请求模型"""
    text: str
    stream: bool = True

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    model: str
    api_base: str

# API路由
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回前端页面"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    translator = get_translator()
    return {
        "status": "healthy",
        "model": translator.model,
        "api_base": translator.base_url
    }

@app.post("/translate/pm-to-dev")
async def translate_product_to_dev(request: TranslationRequest):
    """产品经理→开发工程师翻译"""
    try:
        if request.stream:
            # 流式响应 - 创建异步生成器函数
            async def generate_stream():
                # 注意：这里需要正确调用translate_pm_to_dev
                # 它应该返回一个异步生成器
                translator = get_translator()
                async for chunk in translator.translate_pm_to_dev_stream(request.text):
                    yield chunk
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # 非流式响应
            translator = get_translator()
            result = []
            async for chunk in translator.translate_pm_to_dev_stream(request.text):
                result.append(chunk)
            return {"translation": "".join(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"翻译失败: {str(e)}")

@app.post("/translate/dev-to-pm")
async def translate_dev_to_product(request: TranslationRequest):
    """开发工程师→产品经理翻译"""
    try:
        if request.stream:
            # 流式响应
            async def generate_stream():
                translator = get_translator()
                async for chunk in translator.translate_dev_to_pm_stream(request.text):
                    yield chunk
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # 非流式响应
            translator = get_translator()
            result = []
            async for chunk in translator.translate_dev_to_pm_stream(request.text):
                result.append(chunk)
            return {"translation": "".join(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"翻译失败: {str(e)}")

@app.get("/api/info")
async def get_api_info():
    """获取API信息"""
    return {
        "name": "沟通翻译助手",
        "version": "1.0.0",
        "description": "产品经理和开发工程师之间的沟通翻译工具",
        "endpoints": {
            "product_to_dev": "/translate/pm-to-dev",
            "dev_to_product": "/translate/dev-to-pm",
            "health": "/health"
        },
        "features": ["流式输出", "双向翻译", "实时响应"]
    }

# 启动时的事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    print("🚀 沟通翻译助手启动中...")
    
    # 检查环境变量
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  警告: DEEPSEEK_API_KEY环境变量未设置")
        print("   请在.env文件中设置DEEPSEEK_API_KEY")
    
    # 测试API连接
    translator = get_translator()
    print(f"🔗 正在测试DeepSeek API连接...")
    
    import asyncio
    try:
        connected = await translator.test_connection()
        if connected:
            print("✅ DeepSeek API连接成功")
        else:
            print("❌ DeepSeek API连接失败")
    except Exception as e:
        print(f"❌ API连接测试异常: {str(e)}")
    
    print("✅ 应用启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    print("🛑 正在关闭应用...")
    from app.ai_engine import cleanup
    await cleanup()
    print("✅ 应用已关闭")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )