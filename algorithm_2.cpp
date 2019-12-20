#include<iostream>
#include<vector>
#include<tr1/unordered_map>
#include<map>
#include<algorithm>
#include<math.h>
#include<stack>
#include<queue>
#include<set>
#include<climits>
#include<numeric>
#include<time.h>
#include<list>

using namespace std;
using namespace std::tr1;

/*
693. 交替位二进制数

给定一个正整数，检查他是否为交替位二进制数：换句话说，就是他的二进制数相邻的两个位数永不相等。

示例 1:
输入: 5
输出: True
解释:5的二进制数是: 101
*/
bool hasAlternatingBits(int n) {
	int m=n>>1;

	int i=0;
	int l=n;
	int max=1;
	while(l != 0){
		l=l>>1;
		max=max<<1;
		++i;
	}
	return (n^m)==(max-1);
}

/*
696. 计数二进制子串

给定一个字符串 s，计算具有相同数量0和1的非空(连续)子字符串的数量，
并且这些子字符串中的所有0和所有1都是组合在一起的。
重复出现的子串要计算它们出现的次数。

示例 1 :
输入: "00110011"
输出: 6
解释: 有6个子串具有相同数量的连续1和0：“0011”，“01”，“1100”，“10”，“0011” 和 “01”。
请注意，一些重复出现的子串要计算它们出现的次数。
另外，“00110011”不是有效的子串，因为所有的0（和1）没有组合在一起。
*/
int countBinarySubstrings(string s) {
       int* ar = new int[50000];

        int count = 1;

        int i = 0,j;

        for (j = 1;j < (int)s.size();j++)

        {

            if (s[j] == s[j-1])

                count++;

            else {

                ar[i++] = count;

                count = 1;

            }

        }

        ar[i++] = count;

        int sum = 0;

        while (--i)

        {

            sum += min(ar[i],ar[i-1]);

        }

        delete []ar;

        return sum;
}

/*
697. 数组的度

给定一个非空且只包含非负数的整数数组 nums, 数组的度的定义是指数组里任一元素出现频数的最大值。
你的任务是找到与 nums 拥有相同大小的度的最短连续子数组，返回其长度。

示例 1:
输入: [1, 2, 2, 3, 1]
输出: 2
解释: 
输入数组的度是2，因为元素1和2的出现频数最大，均为2.
连续子数组里面拥有相同度的有如下所示:
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
最短连续子数组[2, 2]的长度为2，所以返回2.
*/
int findShortestSubArray(vector<int>& nums) {
	map<int,std::vector<int>> tmp;
	for(size_t i=0;i<nums.size();++i){
		tmp[nums[i]].push_back(i);
	}

	int n=0;
	int res=INT_MAX;
	for(auto it=tmp.begin(); it!=tmp.end();++it){
		int m=(int)it->second.size();
		if(n<m){
			n=m;
			res=it->second[m-1]-it->second[0]+1;
		}
		if(m==n){
			res=min(res,it->second[m-1]-it->second[0]+1);
		}
	}

	return res;
}

/*
700. 二叉搜索树中的搜索

给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 
返回以该节点为根的子树。 如果节点不存在，则返回 NULL。

例如，
给定二叉搜索树:
        4
       / \
      2   7
     / \
    1   3
和值: 2
你应该返回如下子树:
      2     
     / \   
    1   3
在上述示例中，如果要找的值是 5，但因为没有节点值为 5，我们应该返回 NULL。
*/
struct TreeNode {
	int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
TreeNode* searchBST(TreeNode* root, int val) {
	if(root==NULL){
		return NULL;
	}else{

		if(root->val==val){
			return root;
		}else if(root->val>val){
			return searchBST(root->left,val);
		}else{
			return searchBST(root->left,val);
		}
	}
}

/*
703. 数据流中的第K大元素

设计一个找到数据流中第K大元素的类（class）。注意是排序后的第K大元素，不是第K个不同的元素。
你的 KthLargest 类需要一个同时接收整数 k 和整数数组nums 的构造器，它包含数据流中的初始元素。
每次调用 KthLargest.add，返回当前数据流中第K大的元素。

示例:
int k = 3;
int[] arr = [4,5,8,2];
KthLargest kthLargest = new KthLargest(3, arr);
kthLargest.add(3);   // returns 4
kthLargest.add(5);   // returns 5
kthLargest.add(10);  // returns 5
kthLargest.add(9);   // returns 8
kthLargest.add(4);   // returns 8。
*/
class KthLargest {
public:
	map<int, int> signs;
	int mapSize = 0;
	int k = 0;
	KthLargest(int k, vector<int>& nums) {
		this->k = k;
		for (int i = 0; i < (int)nums.size(); i++) {
			add(nums[i]);
		}
	}

	int add(int val) {
		if (mapSize < k) {
			signs[val]++;
			mapSize++;
		}
		else if(val > signs.begin()->first) {
			int count = signs.begin()->second;
			if (count == 1) {
				signs.erase(signs.begin());
			}
			else {
				signs.begin()->second = count - 1;
			}
			signs[val]++;
		}
		return signs.begin()->first;
	}
};

/*
704. 二分查找

给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

示例 1:
输入: nums = [-1,0,3,5,9,12], target = 9
输出: 4
解释: 9 出现在 nums 中并且下标为 4
*/
int search(vector<int>& nums, int target) {
	int start=0,end=nums.size()-1;
	while(start <= end){
		int mid=(start+end)/2;
		if(nums[mid]== target){
			return mid;
		}else if(nums[mid]<target){
			start=mid+1;
		}else{
			end=mid-1;
		}
	}

	return -1;
}

/*
705. 设计哈希集合

不使用任何内建的哈希表库设计一个哈希集合
具体地说，你的设计应该包含以下的功能
	add(value)：向哈希集合中插入一个值。
	contains(value) ：返回哈希集合中是否存在这个值。
	remove(value)：将给定值从哈希集合中删除。如果哈希集合中没有这个值，什么也不做。

示例:
MyHashSet hashSet = new MyHashSet();
hashSet.add(1);         
hashSet.add(2);         
hashSet.contains(1);    // 返回 true
hashSet.contains(3);    // 返回 false (未找到)
hashSet.add(2);          
hashSet.contains(2);    // 返回 true
hashSet.remove(2);          
hashSet.contains(2);    // 返回  false (已经被删除)
*/
class MyHashSet {
public:
    /** Initialize your data structure here. */
    int N=1001;
    MyHashSet() {
        hashtable = vector<list<int>>(N);
    }
    
    void add(int key) { 
        int mod = key% N;
        if(contains(key))
            return;
        hashtable[mod].push_front(key);
    }
    
    void remove(int key) { 
        int mod = key% N;
        if(!contains(key))return;
        hashtable[mod].remove(key);
    }
    
    /** Returns true if this set contains the specified element */
    bool contains(int key) {
        int mod = key% N;
        if( find(hashtable[mod].begin(),hashtable[mod].end(),key ) != hashtable[mod].end()  )
            return true;
        return false;
    }

private:
    vector<list<int>> hashtable; 
};

/*
706. 设计哈希映射

不使用任何内建的哈希表库设计一个哈希映射
具体地说，你的设计应该包含以下的功能
	put(key, value)：向哈希映射中插入(键,值)的数值对。如果键对应的值已经存在，更新这个值。
	get(key)：返回给定的键所对应的值，如果映射中不包含这个键，返回-1。
	remove(key)：如果映射中存在这个键，删除这个数值对。

示例：
MyHashMap hashMap = new MyHashMap();
hashMap.put(1, 1);          
hashMap.put(2, 2);         
hashMap.get(1);            // 返回 1
hashMap.get(3);            // 返回 -1 (未找到)
hashMap.put(2, 1);         // 更新已有的值
hashMap.get(2);            // 返回 1 
hashMap.remove(2);         // 删除键为2的数据
hashMap.get(2);            // 返回 -1 (未找到) 
*/
class MyHashMap {
private:
    struct node {
        int value;
        int key;
        node* next;
        node(int k, int val): key(k), value(val), next(NULL) {};
    };
    vector<node*> dict;
    int size = 10000;
public:
    /** Initialize your data structure here. */
    MyHashMap() {
        dict = vector<node*>(size, new node(-1, -1));
    }
    
    /** value will always be non-negative. */
    void put(int key, int value) {
        int index = key % size;
        node* temp = dict[index];
        node* last_node;
        while(temp != NULL) {
            if (temp->key == key) {
                temp->value = value;
                return;
            }
            last_node = temp;
            temp = temp->next;
        }
        node* cur = new node(key, value);
        last_node->next = cur;
    }
    
    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    int get(int key) {
        int index = key % size;
        node* temp = dict[index];
        while(temp != NULL) {
            if (temp->key == key) {
                return temp->value;
            }
            temp = temp->next;
        }
        return -1;
    }
    
    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    void remove(int key) {
        int index = key % size;
        node* temp = dict[index];
        node* last_node = temp;;
        while(temp != NULL) {
            if (temp->key == key) {
                last_node->next = temp->next;
                return;
            }
            last_node = temp;
            temp = temp->next;
        }
    }
};

/*
709. 转换成小写字母

实现函数 ToLowerCase()，该函数接收一个字符串参数 str，并将该字符串中的大写字母转换成小写字母，之后返回新的字符串。

示例 1：
输入: "Hello" 输出: "hello"
*/
string toLowerCase(string str) {
	for(auto it=str.begin();it!=str.end();++it){
		if(*it>'A' && *it<'Z'){
			*it=*it-'A'+'a';
		}
	}

	return str;
}

/*
717. 1比特与2比特字符

有两种特殊字符。第一种字符可以用一比特0来表示。第二种字符可以用两比特(10 或 11)来表示。
现给一个由若干比特组成的字符串。问最后一个字符是否必定为一个一比特字符。给定的字符串总是由0结束。

示例 1:
输入: 
bits = [1, 0, 0]
输出: True
解释: 唯一的编码方式是一个两比特字符和一个一比特字符。所以最后一个字符是一比特字符。
*/
bool isOneBitCharacter(vector<int>& bits) {
	return true;
}

int main(){
	cout<<hasAlternatingBits(1431655764)<<endl;


	return 0;
}