# Deep Learning: Framework

Ahmed Furkan Ozkalay, Aras Mumcuyan

### Quick start
     
    # Dataset
    x: torch.FloatTensor  # N x feature_number
    y: torch.FloatTensor  # N x class_number
    

    # Defining a Sequential model 
    model = Sequential()
    model.add(Linear(out=25, input_size=x.shape[1], activation='relu'))
    model.add(Linear(out=25, activation='relu'))
    model.add(Linear(out=25, activation='relu'))
    model.add(Linear(out=y.shape[1], activation='sigmoid'))
    
    # Setting loss from modul
    model.loss = LossMSE()
    
    # Defining an optimizer with following parameters 
    opt = SGD(lr=0.01, momentum_coef=0.1, weight_decay=0.2)
    
    # traing model with optimizer's train function
    report = opt.train(model, x_train, y_train, num_of_epocs=1000, batch_size=128, val_split=0.2, verbose=1)
    
    # evaluating model
    test_accuracy, test_loss = model.evaluate(x_test, y_test)
    
    # predicting test 
    y_preds = model.predict(x_test)
    
### Docker 

    Dockerfile is provided
    
    # to build docker-image: 
        docker build -t dl-framework .
    
    # to run test.py file 
        docker run dl-framework python test.py
    
    # go to localhost:8888/ to run 
    # to run it as a background task 
        docker run --rm -d -v "$(pwd):/app" -p 8888:8888 dl-framework
    
    # to access docker environment terminal
         docker run -v "$(pwd):/app" -it dl-framework /bin/bash
     
 