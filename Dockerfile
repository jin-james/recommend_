FROM tolbkni/bert-as-service
RUN pip install supervisor Flask flask-apidoc Flask-HTTPAuth requests gunicorn flask-restful flask-script scikit-learn keras==2.1.4 numpy jieba gensim --index-url=https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
COPY ./ /app
WORKDIR /app
EXPOSE 5001
CMD ["/usr/local/bin/supervisord", "-c", "/app/supervisord.conf"]
