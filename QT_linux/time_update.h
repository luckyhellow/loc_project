#ifndef TIME_UPDATE_H
#define TIME_UPDATE_H

#include <QTimer>
#include <QObject>

class Time_update : public QObject
{
    Q_OBJECT
public:
    QTimer *timeupdate = new QTimer(this);
    explicit Time_update(QObject *parent = nullptr);
//    void begin_update();
    void begin_update(int hz = 1);
};

#endif // TIME_UPDATE_H
