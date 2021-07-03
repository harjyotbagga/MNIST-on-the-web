# MNIST on the Web
An attempt to predict MNIST handwritten digits from my PyTorch model from the browser (client-side) and not from the server, with the help of [onnx.js](https://onnx.ai/).

## About onnx.js
ONNX.js is a Javascript library for running ONNX models on browsers and on Node.js.  
The Open Neural Network Exchange (ONNX) is an open standard for representing machine learning models. The biggest advantage of ONNX is that it allows interoperability across different open source AI frameworks, which itself offers more flexibility for AI frameworks adoption.

### Advantages of running models on the browser (client-side)
- Faster inference time with small models
- Easy to host & scale models
- Offline Support
- User Privacy

### Disadvantages of running models on the browser (client-side)
- Faster load times
- Faster & consistent inference times with larger models
- Model Privacy

## Installation & Excecution Steps
1. Clone this repository
    ```sh
    git clone https://github.com/harjyotbagga/MNIST-on-the-web.git
    ```
2. Build the docker image
    ```sh
    cd MNIST-on-the-web
    docker build -t harjyotbagga/mnist-web:latest .
    ```
3. Start the project by running the docker image
    ```sh
    docker run -p 80:80 harjyotbagga/mnist-web
    ```

## Live demonstration
https://bugz-mnist.herokuapp.com/

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Issues
For any bugs, feature requests, discussions or comments please open an issue [here](https://github.com/harjyotbagga/MNIST-on-the-web/issues)

<div align="center">

### Made with ❤️
</div>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->