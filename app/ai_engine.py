"""
DeepSeek翻译引擎 - 轻量级HTTP版本
使用aiohttp直接调用DeepSeek API，无OpenAI依赖
"""

import os
import asyncio
import json
import aiohttp
from typing import AsyncGenerator, Dict, Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class DeepSeekTranslator:
    """DeepSeek翻译引擎"""
    
    def __init__(self):
        """初始化翻译引擎"""
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "2000"))
        
        if not self.api_key:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
        
        # 加载提示词模板
        from app.prompts import PM_TO_DEV_PROMPT, DEV_TO_PM_PROMPT
        self.pm_to_dev_prompt = PM_TO_DEV_PROMPT
        self.dev_to_pm_prompt = DEV_TO_PM_PROMPT
        
        # 连接配置
        self.timeout = aiohttp.ClientTimeout(total=60)
        self._session = None
        
        print(f"✅ DeepSeek翻译引擎初始化完成")
        print(f"   模型: {self.model}")
        print(f"   API地址: {self.base_url}")
        
    async def get_session(self) -> aiohttp.ClientSession:
        """获取或创建HTTP会话（连接池）"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,  # 最大连接数
                ttl_dns_cache=300,  # DNS缓存时间
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
                headers={
                    "User-Agent": "CommunicationTranslator/1.0"
                }
            )
        return self._session
    
    async def close(self):
        """关闭HTTP会话"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def _create_request_data(
        self, 
        messages: list, 
        stream: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """创建API请求数据"""
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    
    def _create_messages(self, text: str, direction: str) -> list:
        """根据方向创建消息列表"""
        if direction == "pm-to-dev":
            system_content = """你是一位资深技术架构师，擅长将业务需求转化为技术方案。
            
请确保回复包含以下方面：
1. **技术实现方案** - 具体技术栈、算法、架构设计
2. **数据来源与处理** - 数据采集、清洗、存储方案
3. **性能要求** - 响应时间、并发量、SLA指标
4. **工作量评估** - 人天估算、技术风险点

使用##标题格式进行结构化回复，让开发人员一目了然。"""
            
            user_content = self.pm_to_dev_prompt.format(input=text)
            
        else:  # dev-to-pm
            system_content = """你是一位资深产品经理，擅长将技术方案转化为业务价值。
            
请确保回复包含以下方面：
1. **用户体验影响** - 对用户使用体验的具体改善
2. **业务增长空间** - 能带来的用户增长、收入提升
3. **商业价值** - 成本节省、效率提升、风险降低

使用通俗易懂的语言，避免技术术语，让产品经理快速理解价值。"""
            
            user_content = self.dev_to_pm_prompt.format(input=text)
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    async def _call_deepseek_api(
        self, 
        messages: list, 
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """调用DeepSeek API核心方法"""
        session = await self.get_session()
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream" if stream else "application/json"
        }
        
        data = self._create_request_data(messages, stream)
        
        try:
            async with session.post(url, headers=headers, json=data) as response:
                # 检查HTTP状态码
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = f"API错误 (HTTP {response.status})"
                    
                    try:
                        error_json = json.loads(error_text)
                        if "error" in error_json and "message" in error_json["error"]:
                            error_msg = error_json["error"]["message"]
                    except:
                        pass
                    
                    yield f"【系统错误】{error_msg}"
                    return
                
                if stream:
                    # 处理流式响应
                    buffer = ""
                    async for chunk in response.content.iter_any():
                        if chunk:
                            buffer += chunk.decode('utf-8', errors='ignore')
                            
                            # 按行处理完整的事件
                            lines = buffer.split('\n')
                            for line in lines[:-1]:  # 处理完整的行
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                if line.startswith("data: "):
                                    data_str = line[6:]  # 移除"data: "前缀
                                    
                                    if data_str == "[DONE]":
                                        continue
                                    
                                    try:
                                        data_obj = json.loads(data_str)
                                        if (data_obj.get("choices") and 
                                            len(data_obj["choices"]) > 0):
                                            
                                            delta = data_obj["choices"][0].get("delta", {})
                                            if delta and "content" in delta:
                                                content = delta["content"]
                                                if content:
                                                    yield content
                                                    
                                            # 检查是否有finish_reason
                                            if data_obj["choices"][0].get("finish_reason"):
                                                # 可以在这里处理完成逻辑
                                                pass
                                                
                                    except json.JSONDecodeError as e:
                                        # 记录但不中断流程
                                        print(f"JSON解析错误: {e}, 原始数据: {data_str[:100]}")
                                        continue
                            
                            # 保留最后一行（可能不完整）
                            buffer = lines[-1] if lines else ""
                else:
                    # 处理非流式响应
                    result = await response.json()
                    if (result.get("choices") and 
                        len(result["choices"]) > 0 and
                        result["choices"][0].get("message") and
                        result["choices"][0]["message"].get("content")):
                        
                        content = result["choices"][0]["message"]["content"]
                        if content:
                            yield content
                            
        except aiohttp.ClientConnectorError as e:
            yield f"【网络错误】无法连接到DeepSeek API: {str(e)}"
            yield "请检查：\n1. 网络连接是否正常\n2. API地址是否正确\n3. 防火墙设置"
        except aiohttp.ClientError as e:
            yield f"【HTTP错误】请求失败: {str(e)}"
        except asyncio.TimeoutError:
            yield "【超时错误】API请求超时，请稍后重试"
        except Exception as e:
            yield f"【未知错误】: {str(e)}"
    
    async def translate_pm_to_dev_stream(self, text: str) -> AsyncGenerator[str, None]:
        """产品经理→开发工程师翻译（流式）"""
        if not text or not text.strip():
            yield "请输入有效的产品需求"
            return
        
        messages = self._create_messages(text, "pm-to-dev")
        async for chunk in self._call_deepseek_api(messages, stream=True):
            yield chunk
    
    async def translate_dev_to_pm_stream(self, text: str) -> AsyncGenerator[str, None]:
        """开发工程师→产品经理翻译（流式）"""
        if not text or not text.strip():
            yield "请输入有效的技术方案"
            return
        
        messages = self._create_messages(text, "dev-to-pm")
        async for chunk in self._call_deepseek_api(messages, stream=True):
            yield chunk
    
    async def translate_pm_to_dev(self, text: str, stream: bool = False) -> AsyncGenerator[str, None] | str:
        """产品经理→开发工程师翻译（兼容接口）"""
        if stream:
            return self.translate_pm_to_dev_stream(text)
        else:
            content_parts = []
            async for chunk in self.translate_pm_to_dev_stream(text):
                content_parts.append(chunk)
            return "".join(content_parts)
    
    async def translate_dev_to_pm(self, text: str, stream: bool = False) -> AsyncGenerator[str, None] | str:
        """开发工程师→产品经理翻译（兼容接口）"""
        if stream:
            return self.translate_dev_to_pm_stream(text)
        else:
            content_parts = []
            async for chunk in self.translate_dev_to_pm_stream(text):
                content_parts.append(chunk)
            return "".join(content_parts)
    
    async def test_connection(self) -> bool:
        """测试DeepSeek API连接"""
        try:
            test_messages = [
                {"role": "system", "content": "你是一个测试助手"},
                {"role": "user", "content": "请回复'连接成功'"}
            ]
            
            # 使用非流式请求进行测试
            test_data = self._create_request_data(test_messages, stream=False)
            test_data["max_tokens"] = 10
            
            session = await self.get_session()
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            async with session.post(url, headers=headers, json=test_data) as response:
                if response.status == 200:
                    result = await response.json()
                    return True
                else:
                    print(f"测试失败: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            print(f"连接测试异常: {str(e)}")
            return False
        finally:
            await self.close()

# 全局单例实例
_translator_instance = None

def get_translator() -> DeepSeekTranslator:
    """获取翻译器实例（单例模式）"""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = DeepSeekTranslator()
    return _translator_instance

async def translate_pm_to_dev(text: str, stream: bool = True):
    """产品→开发翻译接口"""
    translator = get_translator()
    return await translator.translate_pm_to_dev(text, stream)

async def translate_dev_to_pm(text: str, stream: bool = True):
    """开发→产品翻译接口"""
    translator = get_translator()
    return await translator.translate_dev_to_pm(text, stream)

async def close_translator():
    """关闭翻译器"""
    global _translator_instance
    if _translator_instance:
        await _translator_instance.close()
        _translator_instance = None

# 清理函数
async def cleanup():
    """清理资源"""
    await close_translator()

# 同步清理函数（供atexit使用）
def sync_cleanup():
    """同步清理资源"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(cleanup())
        loop.close()
    except:
        pass  # 忽略清理时的错误

# 注册清理函数
import atexit
atexit.register(sync_cleanup)