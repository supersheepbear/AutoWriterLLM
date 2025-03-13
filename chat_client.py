import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chat_client.log')
    ]
)

logger = logging.getLogger(__name__)

class LocalChatClient:
    """本地 OpenAI 兼容 API 客户端.

    这个类提供了与本地运行的 OpenAI 兼容 API 服务器交互的功能。

    Attributes:
        base_url (str): API 基础URL
        api_key (str): API 认证令牌
        client (OpenAI): OpenAI 客户端实例

    Examples:
        >>> client = LocalChatClient(api_key="your-token")
        >>> response = client.chat("你好，请介绍一下自己")
        >>> print(response)
    """

    def __init__(self, api_key: str, base_url: str = "http://localhost:3000/v1") -> None:
        """初始化聊天客户端.

        Args:
            api_key (str): API 认证令牌
            base_url (str, optional): API 基础URL. 默认为 "http://localhost:3000/v1"

        Raises:
            ValueError: 如果 api_key 为空
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        
        self.base_url = base_url
        self.api_key = api_key
        
        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
            logger.info("Successfully initialized OpenAI client")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def chat(
        self, 
        message: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """发送聊天消息并获取响应.

        Args:
            message (str): 用户输入的消息
            model (str, optional): 使用的模型名称. 默认为 "gpt-3.5-turbo"
            temperature (float, optional): 响应的随机性. 默认为 0.7
            max_tokens (int, optional): 最大响应令牌数. 默认为 None

        Returns:
            str: API 的响应文本

        Raises:
            Exception: 当 API 调用失败时抛出异常

        Examples:
            >>> client = LocalChatClient(api_key="your-token")
            >>> response = client.chat("Python中如何处理异常？")
            >>> print(response)
        """
        try:
            logger.info(f"Sending message: {message[:50]}...")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            logger.info(f"Received response: {result[:50]}...")
            return result

        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
            raise

def main():
    """主函数用于测试."""
    # 使用示例
    try:
        client = LocalChatClient(api_key="your-workos-cursor-session-token")
        response = client.chat("你好，请介绍一下自己")
        print(f"Response: {response}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 