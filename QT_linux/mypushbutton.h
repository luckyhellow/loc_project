#ifndef MYPUSHBUTTON_H
#define MYPUSHBUTTON_H

#include <QPushButton>
#include <QPropertyAnimation>

class MyPushButton : public QPushButton
{
    Q_OBJECT
public:
    //explicit MyPushButton(QWidget *parent = nullptr);

    MyPushButton(QString normalImg, QString pressImg = "");

    QString normalImgPath;
    QString pressImgPath;

    void tik();
    void tok();
signals:

};

#endif // MYPUSHBUTTON_H
