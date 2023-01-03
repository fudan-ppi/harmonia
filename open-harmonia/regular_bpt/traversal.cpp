
#include "cfb.h"

using namespace std;

typedef struct queue_t{
    node *head;
    node *tail;

    void enqueue(node *n){
	    if (head == NULL) {
		    head = n;
		    head->next = NULL;
            tail = head;
	    }
	    else {
			tail->next = n;
		    n->next = NULL;
            tail = n;
	    }

    }

    node *dequeue() {
        node *r = head;
        head = head->next; 
        return r;
    }
    
    bool isEmpty(){
        return (head==NULL);
    }

} queue_t;

void traverse_transform(CFB *cfb, node *n){
    
    int internal_node_num = 0;
    int node_num = 0;
    cfb->prefix[0] = 1;
    queue_t q; 
    q.enqueue(n);
    while (!q.isEmpty()) {
        n = q.dequeue();
        node_num ++;
        kArr_t *k_arr = cfb->key_section->getElement();
        cArr_t *c_arr = cfb->pointer_section->getElement(); //space for child array.    store nothing in interal node except update, store record pointer(offset) in leaf node.          

        for (int i=0; i<n->num_keys;i++) {
            k_arr->key[i] = n->keys[i];
        }
        for (int i=n->num_keys; i<order;i++) {
            k_arr->key[i] = Max_Key; 
        }
		//printf("%lx %d\n", (unsigned long)n, height(n));

        if (!n->is_leaf) {
            cfb->prefix[internal_node_num+1] =  cfb->prefix[internal_node_num] + (n->num_keys+1);
            
            internal_node_num++;


            for (int i=0; i<n->num_keys+1;i++){
                q.enqueue((node *)(n->pointers[i]));
            }
        }
        else{
            for (int i=0; i<n->num_keys;i++) {
                record *r = cfb->record_section->getElement();
                c_arr->record[i] = cfb->record_section->indexAtAddress(r);
                *r = *((record *)(n->pointers[i]));
                //cout << (r->value) <<endl;
            }
        }
    }
    cfb->internal_node_num = internal_node_num;
    cfb->node_num = node_num;
}


CFB *generateCFB(node *root){
    if (order != DEFAULT_ORDER) {cout<<"order wrong!"<<endl;exit(0);}
    CFB* cfb = new CFB();


    cfb->prefix = (int *)malloc(PREFIX_DEFAULT_LENGTH * sizeof(int));
    
    node *node  = root;
    traverse_transform(cfb, node);
    if (cfb->internal_node_num > PREFIX_DEFAULT_LENGTH) {
        cout<<"internal node num"<<cfb->internal_node_num<<" PREFIX_LENGTH"<<PREFIX_DEFAULT_LENGTH<<endl;
        exit(0);
    }
    cfb->height = height(root);
    //for (int i=0;i<cfb->internal_node_num;i++)
    //    cout<<cfb->prefix[i]<<endl;

    return cfb;
}
void traverse_transform_RB(RB *rb, node *n){
    int node_num = generate_time;
    rb->rootIdx = n->generate_time; 
    for (int i=0;i<node_num;i++) {
        rb->key_section->getElement();
        rb->pointer_section->getElement();
    }
    
    //records' order do not influence GPU, so records are in bfs fashion.

    queue_t q;
    q.enqueue(n);
    
    while (!q.isEmpty()){
        n = q.dequeue();
        int idx = n->generate_time;
        kArr_t *k_arr = rb->key_section->elementAtIdx(idx);
        cArr_t *c_arr = rb->pointer_section->elementAtIdx(idx);

        for (int i=0; i<n->num_keys; i++) {
            k_arr->key[i] = n->keys[i];
        }
        for (int i=n->num_keys; i<order; i++) {
            k_arr->key[i] = Max_Key;
        }
        if (!n->is_leaf) {
            for (int i=0; i<n->num_keys+1; i++) {
                node * next = (node*)n->pointers[i];
                c_arr->child[i] = next->generate_time;
                q.enqueue(next);
            }

        }
        else{
            for (int i=0;i<n->num_keys;i++){
                record *r = rb->record_section->getElement();
                c_arr->record[i] = rb->record_section->indexAtAddress(r);
                *r = *((record *)n->pointers[i]);
            }
        }

    }
    rb->node_num = node_num;

}
RB *generateRB(node *root){
    if (order != DEFAULT_ORDER) {cout<<"order wrong!"<<endl;exit(0);}
    RB* rb = new RB();

    traverse_transform_RB(rb, root);
    rb->height = height(root);
    return rb;
}
