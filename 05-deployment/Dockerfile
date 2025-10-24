FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

# 复制项目依赖配置
COPY pyproject.toml .

# 安装依赖（uv 是这个基础镜像内置的包管理器）
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn pandas numpy

# 复制应用脚本
COPY fastapi_app.py .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]

