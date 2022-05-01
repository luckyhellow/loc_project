#ifndef QLABEL_LIST_H
#define QLABEL_LIST_H
#include <QLabel>
typedef struct Node {
    QLabel * label;
    Node* next = NULL;
} Node;
typedef struct Node* labelList;
bool insert_intail(labelList& head,QLabel* label);
void clear(labelList& head);
class Qlabel_List
{
public:
    Qlabel_List();
};

#endif // QLABEL_LIST_H
