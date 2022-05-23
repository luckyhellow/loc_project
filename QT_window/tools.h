#ifndef TOOLS_H
#define TOOLS_H
#include <QLabel>
#include <string>
#include <time.h>
using namespace std;

struct TIME{
    tm time_s;
    int time_ms;
    friend bool operator<=(TIME time1,TIME time2){
        time_t t1 =  mktime(&time1.time_s);
        time_t t2 =  mktime(&time2.time_s);
        if(t1<t2) return true;
        else if(t1==t2){
            return time1.time_ms<=time2.time_ms;
        }
        else return false;
    }
    friend bool operator<(TIME time1,TIME time2){
        time_t t1 =  mktime(&time1.time_s);
        time_t t2 =  mktime(&time2.time_s);
        if(t1<t2) return true;
        else if(t1==t2){
            return time1.time_ms<time2.time_ms;
        }
        else return false;
    }
    friend bool operator>=(TIME time1,TIME time2){
        time_t t1 =  mktime(&time1.time_s);
        time_t t2 =  mktime(&time2.time_s);
        if(t1>t2) return true;
        else if(t1==t2){
            return time1.time_ms>=time2.time_ms;
        }
        else return false;
    }
    friend bool operator>(TIME time1,TIME time2){
        time_t t1 =  mktime(&time1.time_s);
        time_t t2 =  mktime(&time2.time_s);
        if(t1>t2) return true;
        else if(t1==t2){
            return time1.time_ms>time2.time_ms;
        }
        else return false;
    }
};

struct Struct_XY{
    string label;
    TIME time;
    double x;
    double y;
};

struct Node {
    QLabel * qlabel;
    Node* next = NULL;
};

typedef struct Node* QlabelList;

void insert(QlabelList& head,QLabel* qlabel);
void clean(QlabelList& head);
void hideqlabels(QlabelList& head);
void showqlabels(QlabelList& head);

#endif // TOOLS_H
