#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>

#define WINDOW_LENGTH 1500
#define WINDOW_WIDTH 1000

#define MAX_LENTH 1000
#define MAX_WIDTH 800

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    scale = min(WINDOW_LENGTH*0.8/MAX_LENTH,WINDOW_WIDTH*0.7/MAX_WIDTH);
    ui->setupUi(this);
    setWindowIcon(QIcon(":/Image/Icon.jpg"));
    //重置窗口大小
    resize(WINDOW_LENGTH,WINDOW_WIDTH);
    //固定大小
    setFixedSize(WINDOW_LENGTH,WINDOW_WIDTH);
    //设置窗口标题
    setWindowTitle("(lucky_demo)locate app");


      // //////////
     //   菜单   //   start
    // //////////


    //创建菜单栏
    QMenuBar* bar = menuBar();
    //菜单栏放入窗口
    setMenuBar(bar);
    //创建菜单
    QMenu *fileMenu = bar->addMenu("功能");
    QMenu *editMenu = bar->addMenu("编辑");

    fileMenu->addAction("尚未完善");
    fileMenu->addSeparator();
    fileMenu->addAction("尚未完善");
    editMenu->addAction("尚未完善");

      // //////////
     //   菜单   //   end
    // //////////

    //create button group to choose labels
    QGroupBox *labelGroup = new QGroupBox(this);
    labelGroup->setTitle("choose labels");
    labelGroup->setGeometry(WINDOW_LENGTH*0.01, WINDOW_WIDTH*0.05, WINDOW_LENGTH*0.18, WINDOW_WIDTH*0.5);

    QVBoxLayout *labelLayout = new QVBoxLayout(labelGroup);

    this->TU = new Time_update(this);//Create Qtime

    //输入框
    QLineEdit* lineedit = new QLineEdit();
    QIntValidator *intValidator = new QIntValidator;
    //setting the legal range
    intValidator->setRange(1, 1000);
    lineedit->setValidator(intValidator);
    lineedit->setParent(this);
    lineedit->resize(WINDOW_LENGTH*0.1,WINDOW_WIDTH*0.035);
    lineedit->move(WINDOW_LENGTH*0.1,WINDOW_WIDTH*0.85);
    lineedit->show();

    //框
    QLabel* backlabel = new QLabel(this);
    backlabel->resize(MAX_LENTH*scale,MAX_WIDTH*scale);
    backlabel->setStyleSheet("border: 5px solid black;");
    backlabel->move(WINDOW_LENGTH*0.2,WINDOW_WIDTH*0.05);
    backlabel->show();

    //creat a new instance of "udprecv" to recv the data
    udp = new UDPrecv();

    connect(TU->timeupdate, &QTimer::timeout,[=](){//fluent to show
        //get the location of xy now
        struct_xy = udp->getxy();
        //creat a qlabel and use it show the xy's location
        //watch out Memory leak!
        if(struct_xy.label!="" && hashmap.count(struct_xy.label)==0){
            //this label is received at the first time
            //creat hashmap
            string strconnect = struct_xy.label;
            hashmap[strconnect] = NULL;

            //generate a new picture with random color
            QImage image;
            int r = rand()%128+rand()%128;
            int g = rand()%128+rand()%128;
            int b = rand()%128+rand()%128;
            image.load(":/Image/point.png");
            int w=image.width();
            int h=image.height();
            for(int i=0;i<h;i++){
                for(int j=0;j<w;j++){
                    image.setPixel(j,i,qRgb(r,g,b));
                }
            }
            hashpix[strconnect] = QPixmap(QPixmap::fromImage(image));

            //create button
            checkbox = new QCheckBox(QString::fromStdString(strconnect), labelGroup);
            //set the button's state to having been chose
            checkbox->setChecked(true);
            hashchoose[strconnect] = true;
            //layout it
            labelLayout->addWidget(checkbox);
            labelGroup->setLayout(labelLayout);
            //add to vector
            qcheckboxs.push_back(checkbox);
            //when the state of buttons change, show or hide
            connect(checkbox, &QCheckBox::toggled, [=](bool isChecked){
                    if (isChecked){
//                        qDebug() << " true ";
//                        cout<<strconnect<<endl;
                        hashchoose[strconnect] = true;
                        showqlabels(hashmap[strconnect]);
                    }
                    else{
//                        qDebug() << " false ";
//                        cout<<strconnect<<endl;
                        hashchoose[strconnect] = false;
                        hideqlabels(hashmap[strconnect]);
                    }
                });
        }
        qlabel = new QLabel(this);
        qlabel->setPixmap(hashpix[struct_xy.label]);
        qlabel->setAlignment(Qt::AlignCenter);
        qlabel->resize(WINDOW_LENGTH*0.01,WINDOW_LENGTH*0.01);
        qlabel->move(WINDOW_LENGTH*0.2+struct_xy.x*scale,struct_xy.y*scale+WINDOW_WIDTH*0.05);
        if(hashchoose[struct_xy.label]) qlabel->show();
        //add to list inorder to realize the function of deleting labels
        insert(hashmap[struct_xy.label],qlabel);
    });

    //create button to control start or not
    //and connect it with HZ of showing label
    MyPushButton *startButton = new MyPushButton(":/Image/startButton.png");
    startButton->setParent(this);
    startButton->move(WINDOW_LENGTH*0.25,WINDOW_WIDTH*0.8);
    connect(startButton,&MyPushButton::clicked,[=](){
        //a simple animation effect
        startButton->tik();
        startButton->tok();
        if(lineedit->text()==""){
            TU->begin_update();
        }
        else {
            hz = lineedit->text().toInt();
            TU->begin_update(hz);
        }
    });

    //create button to clear the labels we have created
    MyPushButton *clearButton = new MyPushButton(":/Image/clearButton.png");
    clearButton->setParent(this);
    clearButton->move(WINDOW_LENGTH*0.6,WINDOW_WIDTH*0.8);
    connect(clearButton,&MyPushButton::clicked,[=](){
        //a simple animation effect
        clearButton->tik();
        clearButton->tok();
        //clear the show and delete qlabels we create with "new"
        //watch out memory leak
        for(const auto &keyvalue:hashmap){
//            cout<<"label: "<<keyvalue.first<<endl;
            clean(hashmap[keyvalue.first]);
        }
    });

}

void MainWindow::paintEvent(QPaintEvent *)
{
    //paint the backgroud
    QPainter painter(this);
    QPixmap pix;
    pix.load(":/Image/floor.jpg");
    painter.drawPixmap(0,0,this->width(),this->height(),pix);

    pix.load(":/Image/white.jpg");
    painter.drawPixmap(WINDOW_LENGTH*0.2,0,WINDOW_LENGTH*0.8,WINDOW_WIDTH*0.75,pix);

    pix.load(":/Image/background.jpg");
    painter.drawPixmap(0,WINDOW_WIDTH*0.75,WINDOW_LENGTH,WINDOW_WIDTH*0.25,pix);
}


MainWindow::~MainWindow()
{
    delete ui;
    for(auto qcheckbox:qcheckboxs){
        delete qcheckbox;
    }
}

