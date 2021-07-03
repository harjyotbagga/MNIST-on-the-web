  
FROM nginx:alpine
COPY . /usr/share/nginx/html

# To Build Image: 
# docker build -t harjyotbagga/mnist-web:latest .
# To Run Image:
# docker run -p 80:80 harjyotbagga/mnist-web