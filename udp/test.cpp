#include <time.h>
#include <iostream>
using namespace std;

struct TIME{
    tm time_s;
    int time_ms;
};

bool operator<=(TIME time1,TIME time2){
    time_t t1 =  mktime(&time1.time_s);
    time_t t2 =  mktime(&time2.time_s);
    if(t1<t2) return true;
    else if(t1==t2){
        return time1.time_ms<=time2.time_ms;
    }
    else return false;
}
bool operator<(TIME time1,TIME time2){
    time_t t1 =  mktime(&time1.time_s);
    time_t t2 =  mktime(&time2.time_s);
    if(t1<t2) return true;
    else if(t1==t2){
        return time1.time_ms<time2.time_ms;
    }
    else return false;
}
bool operator>=(TIME time1,TIME time2){
    time_t t1 =  mktime(&time1.time_s);
    time_t t2 =  mktime(&time2.time_s);
    if(t1>t2) return true;
    else if(t1==t2){
        return time1.time_ms>=time2.time_ms;
    }
    else return false;
}
bool operator>(TIME time1,TIME time2){
    time_t t1 =  mktime(&time1.time_s);
    time_t t2 =  mktime(&time2.time_s);
    if(t1>t2) return true;
    else if(t1==t2){
        return time1.time_ms>time2.time_ms;
    }
    else return false;
}
int main(){
    time_t rawtime = 0;
    tm tm1 = *localtime( &rawtime );
    TIME time1;
    time1.time_ms = 0;
    time1.time_s = tm1;
    rawtime = 2;
    tm tm2 = *localtime( &rawtime );
    TIME time2;
    time2.time_ms = 0;
    time2.time_s = tm2;
    cout<<(time1<=time2)<<endl;;
    return 0;
}