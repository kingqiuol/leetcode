#include<iostream>
#include<fstream>
#include<vector>
#include<tr1/unordered_map>
#include<tr1/unordered_set>
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
    	int key;
        int value;
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
	        int len=bits.size();
        if(bits[len-1]==1) return false;  //最后一位是1,false
        for(int i=0;i<len;i++){
            if(i==len-1) return true;       //最后一位是0，就true（这里由于已经排除了最后一位是1的情况，只要可以到最后一位，就是对的）
            if(bits[i]==1&&i<len-1) i++; //其实10和11是一样的，都占两位，根本没有其他情况
        }
        return false;
}

/*
720. 词典中最长的单词

给出一个字符串数组words组成的一本英语词典。从中找出最长的一个单词，该单词是由words词典中其他单词逐步添加一个字母组成。
若其中有多个可行的答案，则返回答案中字典序最小的单词。若无答案，则返回空字符串。

示例 1:
输入: 
words = ["w","wo","wor","worl", "world"]
输出: "world"
解释: 
单词"world"可由"w", "wo", "wor", 和 "worl"添加一个字母组成。
*/
string longestWord(vector<string>& words) {
	        sort(words.begin(),words.end());
        set<string> tmp;
        tmp.insert("");
        string res="";
        for(auto word:words)
        {
            auto cc=word.substr(0,word.length()-1);
            if(tmp.find(cc)!=tmp.end())
            {
                tmp.insert(word);
                if(word.length()>res.length())
                {
                    res=word;
                }
            }
        }
        return res;
}

/*
724. 寻找数组的中心索引

给定一个整数类型的数组 nums，请编写一个能够返回数组“中心索引”的方法。
我们是这样定义数组中心索引的：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。
如果数组不存在中心索引，那么我们应该返回 -1。如果数组有多个中心索引，那么我们应该返回最靠近左边的那一个。

示例 1:
输入: 
nums = [1, 7, 3, 6, 5, 6]
输出: 3
解释: 索引3 (nums[3] = 6) 的左侧数之和(1 + 7 + 3 = 11)，与右侧数之和(5 + 6 = 11)相等。
同时, 3 也是第一个符合要求的中心索引。
*/
int pivotIndex(vector<int>& nums) { 
        if (nums.size() == 0) return -1;
        int left=0;
        int right = 0;
        for (size_t i=1;i<nums.size();i++){
            right += nums[i];
        }
        for (size_t i=0;i<nums.size() -1 ;i++){
            if (right == left) return i;
            right = right - nums[i+1];
            left += nums[i];
        }
        if (left==right) return nums.size() - 1;
        else return -1;
}

/*
728. 自除数

自除数 是指可以被它包含的每一位数除尽的数。
例如，128 是一个自除数，因为 128 % 1 == 0，128 % 2 == 0，128 % 8 == 0。
还有，自除数不允许包含 0 。
给定上边界和下边界数字，输出一个列表，列表的元素是边界（含边界）内所有的自除数。

示例 1：
输入： 
上边界left = 1, 下边界right = 22
输出： [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
*/
vector<int> selfDividingNumbers(int left, int right) {
	std::vector<int> res;
	for(int i=left;i<=right;++i){
		int num=i;
		while(num !=0){	
			
			int tmp=num%10;	
			if(tmp==0 ||(tmp!=0 && i%tmp!=0) ){
			
				break;
			}

			num=num/10;
		}
		
		if(num==0){
			res.push_back(i);
		}
	}

	return res;
}

/*
733. 图像渲染

有一幅以二维整数数组表示的图画，每一个整数表示该图画的像素值大小，数值在 0 到 65535 之间。
给你一个坐标 (sr, sc) 表示图像渲染开始的像素值（行 ，列）和一个新的颜色值 newColor，让你重新上色这幅图像。
为了完成上色工作，从初始坐标开始，记录初始坐标的上下左右四个方向上像素值与初始坐标相同的相连像素点，
接着再记录这四个方向上符合条件的像素点与他们对应四个方向上像素值与初始坐标相同的相连像素点，……，重复该过程。
将所有有记录的像素点的颜色值改为新的颜色值。最后返回经过上色渲染后的图像。

示例 1:
输入: 
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1, sc = 1, newColor = 2
输出: [[2,2,2],[2,2,0],[2,0,1]]
解析: 
在图像的正中间，(坐标(sr,sc)=(1,1)),
在路径上所有符合条件的像素点的颜色都被更改成2。
注意，右下角的像素没有更改为2，
因为它不是在上下左右四个方向上与初始点相连的像素点。
*/
vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
	if(image.size()==0)return image;
        int dx[4]={-1,0,1,0},dy[4]={0,1,0,-1};
        int oldColor=image[sr][sc];//记录旧颜色
        image[sr][sc]=newColor;//变新颜色
        if(oldColor==newColor)return image;//特判一下不然会死循环
        for(int i=0;i<4;i++)
        {
            int x=sr+dx[i],y=sc+dy[i];
            if(x>=0 && x<(int)image.size() && y>=0 &&y<(int)image[0].size()&&image[x][y]==oldColor)
            {
                floodFill(image,x,y,newColor);
            }
        
        }

        return image;
}

/*
744. 寻找比目标字母大的最小字母

给定一个只包含小写字母的有序数组letters 和一个目标字母 target，寻找有序数组里面比目标字母大的最小字母。
数组里字母的顺序是循环的。举个例子，如果目标字母target = 'z' 并且有序数组为 letters = ['a', 'b']，则答案返回 'a'。

示例:
输入:
letters = ["c", "f", "j"]
target = "a"
输出: "c"
*/
char nextGreatestLetter(vector<char>& letters, char target) {
         int l = 0, r = letters.size() - 1;
        if(target >= letters[r] || target < letters[l])  return letters[l];
        while(l + 1 < r){ 
            int mid = l + (r - l)/2;
            if(letters[mid] > target) r = mid;
            else    l = mid;
        }
        return letters[r];
}

/*
746. 使用最小花费爬楼梯

数组的每个索引做为一个阶梯，第 i个阶梯对应着一个非负数的体力花费值 cost[i](索引从0开始)。
每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。
您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。

示例 1:
输入: cost = [10, 15, 20]
输出: 15
解释: 最低花费是从cost[1]开始，然后走两步即可到阶梯顶，一共花费15。
*/
int minCostClimbingStairs(vector<int>& cost) {
	int p=0,q=0;
	int res=0;
	for(size_t i=0;i<cost.size()-1;++i){
		res=min(p+cost[i],q+cost[i+1]);
		p=q;
		q=res;
	}

	return res;
}

/*
747. 至少是其他数字两倍的最大数

在一个给定的数组nums中，总是存在一个最大元素 。
查找数组中的最大元素是否至少是数组中每个其他数字的两倍。
如果是，则返回最大元素的索引，否则返回-1。

示例 1:
输入: nums = [3, 6, 1, 0]
输出: 1
解释: 6是最大的整数, 对于数组中的其他整数,
6大于数组中其他元素的两倍。6的索引是1, 所以我们返回1.
*/
int dominantIndex(vector<int>& nums) {
	if(nums.empty()){
		return -1;
	}

	if(nums.size()==1){
		return 0;
	}

    map<int,int,greater<int>> maps;
	for(size_t i=0;i<nums.size();++i){
		maps.insert(make_pair(nums[i],i));
	}

	auto it=maps.begin();
    auto next=maps.begin();
    ++next;
	if(it->first > 2*next->first){
		return it->second;
	}else{
		return -1;
	}
}

/*
748. 最短完整词

如果单词列表（words）中的一个单词包含牌照（licensePlate）中所有的字母，那么我们称之为完整词。在所有完整词中，
最短的单词我们称之为最短完整词。单词在匹配牌照中的字母时不区分大小写，比如牌照中的 "P" 依然可以匹配单词中的 "p" 字母。
我们保证一定存在一个最短完整词。当有多个单词都符合最短完整词的匹配条件时取单词列表中最靠前的一个。
牌照中可能包含多个相同的字符，比如说：对于牌照 "PP"，单词 "pair" 无法匹配，但是 "supper" 可以匹配。

示例 1：
输入：licensePlate = "1s3 PSt", words = ["step", "steps", "stripe", "stepple"]
输出："steps"
说明：最短完整词应该包括 "s"、"p"、"s" 以及 "t"。
对于 "step" 它只包含一个 "s" 所以它不符合条件。同时在匹配过程中我们忽略牌照中的大小写。
*/
string shortestCompletingWord(string licensePlate, vector<string>& words) {
     map<char,int> temp1;
    for(size_t i=0;i<licensePlate.size();++i){

        if(licensePlate[i]>='A' && licensePlate[i]<='Z'){
            temp1[licensePlate[i]+32]++;
        }else if(licensePlate[i]>='a' && licensePlate[i]<='z'){
            temp1[licensePlate[i]]++;
        }else{
            continue;
        }
    }

    int min=INT_MAX;
    int index=0;
    for(size_t i=0;i<words.size();++i){
        map<char,int> temp2;
        for(size_t j=0;j<words[i].size();++j){
            temp2[words[i][j]]++;
        }

        auto it=temp1.begin();
        for(;it!=temp1.end();++it){
            if(temp2[it->first]<it->second){
                break;
            }
        }

        if(it==temp1.end()){
            if(min>(int)words[i].size()){
                min=(int)words[i].size();
                index=i;
            }
        }
    }

    return words[index];
}

/*
762. 二进制表示中质数个计算置位

给定两个整数 L 和 R ，找到闭区间 [L, R] 范围内，计算置位位数为质数的整数个数。
（注意，计算置位代表二进制表示中1的个数。例如 21 的二进制表示 10101 有 3 个计算置位。还有，1 不是质数。）

示例 1:
输入: L = 6, R = 10
输出: 4
解释:
6 -> 110 (2 个计算置位，2 是质数)
7 -> 111 (3 个计算置位，3 是质数)
9 -> 1001 (2 个计算置位，2 是质数)
10-> 1010 (2 个计算置位，2 是质数)
*/
int countPrimeSetBits(int L, int R) {
	int iCount=0;
    for(int i=L;i<=R;++i){
        int num=i;
        int count=0;
        while(num){
            if(num&1)
                ++count;
            num=num>>1;
        }

        if(count==1)
            continue;
        //两个较小数另外处理
        if(count ==2|| count==3 )
            iCount++ ;
        //不在6的倍数两侧的一定不是质数
        if(count %6!= 1&&count %6!= 5)
            continue;
        int tmp =sqrt( count);
        //在6的倍数两侧的也可能不是质数
        for(int i= 5;i <=tmp; i+=6 )
            if(count %i== 0||count %(i+ 2)==0 )
                return 0 ;
        //排除所有，剩余的是质数
        iCount++;
    }

    return iCount;
}

/*
766. 托普利茨矩阵

如果一个矩阵的每一方向由左上到右下的对角线上具有相同元素，那么这个矩阵是托普利茨矩阵。
给定一个 M x N 的矩阵，当且仅当它是托普利茨矩阵时返回 True。

示例 1:
输入: 
matrix = [
  [1,2,3,4],
  [5,1,2,3],
  [9,5,1,2]
]
输出: True
解释:
在上述矩阵中, 其对角线为:
"[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]"。
各条对角线上的所有元素均相同, 因此答案是True。
*/
bool isToeplitzMatrix(vector<vector<int>>& matrix) {
	       if(matrix.empty()) return true;
        for(int i = 0; i < (int)matrix[0].size(); i++){
            int pos_i = 0, pos_j = i, val = matrix[0][i];
            while(++pos_i < (int)matrix.size() && ++pos_j < (int)matrix[0].size()) {
                if(val != matrix[pos_i][pos_j]) return false;
            }
        }
        for(int i = 0; i < (int)matrix.size(); i++) {
            int pos_i = i, pos_j = 0, val = matrix[i][0];
            while(++pos_i < (int)matrix.size() && ++pos_j < (int)matrix[0].size()) {
                if(val != matrix[pos_i][pos_j]) return false;
            }
        }
        return true;
}

/*
771. 宝石与石头

 给定字符串J 代表石头中宝石的类型，和字符串 S代表你拥有的石头。 
 S 中每个字符代表了一种你拥有的石头的类型，你想知道你拥有的石头中有多少是宝石。
J 中的字母不重复，J 和 S中的所有字符都是字母。字母区分大小写，因此"a"和"A"是不同类型的石头。

示例 1:
输入: J = "aA", S = "aAAbbbb"
输出: 3
*/
int numJewelsInStones(string J, string S) {
	map<char,int> maps;
	for(size_t i=0;i<J.size();++i){
		maps[J[i]]=0;
	}

	int iCount=0;
	for(size_t i=0;i<S.size();i++){
		if(maps.count(S[i])!=0){
			iCount++;
		}
	}

	return iCount;
}

/*
783. 二叉搜索树结点最小距离

给定一个二叉搜索树的根结点 root, 返回树中任意两节点的差的最小值。

示例：
输入: root = [4,2,6,1,3,null,null]
输出: 1
解释:
注意，root是树结点对象(TreeNode object)，而不是数组。
给定的树 [4,2,6,1,3,null,null] 可表示为下图:
          4
        /   \
      2      6
     / \    
    1   3  
最小的差值是 1, 它是节点1和节点2的差值, 也是节点3和节点2的差值。
*/
int mmin = 0x7fffffff;
struct Node {
    int mostL;
    int mostR;
    Node(int x,int y) : mostL(x),mostR(y) {}
};
    Node digui(TreeNode* root){
	if(root->left==NULL&&root->right==NULL){
		Node t = Node(root->val,root->val);
		return t;
	}
	if(root->left==NULL){
		Node rt = digui(root->right);
		mmin = min(mmin, abs(root->val - rt.mostL));
		Node t = Node(root->val,rt.mostR);
		return t;
	}
	if(root->right==NULL){
		Node lt = digui(root->left);
		mmin = min(mmin, abs(root->val - lt.mostR));
		Node t = Node(lt.mostL,root->val);
		return t;
	}
	Node lt = digui(root->left);
	Node rt = digui(root->right);
	mmin = min(mmin,min(abs(root->val - lt.mostR),abs(root->val - rt.mostL)));
	Node t = Node(lt.mostL,rt.mostR);
	return t;
}

 
int minDiffInBST(TreeNode* root) {
    if(root==NULL){
        return 0;
    }
    digui(root);    
    return mmin;
}

/*
784. 字母大小写全排列

给定一个字符串S，通过将字符串S中的每个字母转变大小写，我们可以获得一个新的字符串。返回所有可能得到的字符串集合。

示例:
输入: S = "a1b2"
输出: ["a1b2", "a1B2", "A1b2", "A1B2"]
*/
    void backtrack(string &s, int i, vector<string> &res) {
        if (i ==(int)s.size()) {
            res.push_back(s);
            return;
        }
        backtrack(s, i + 1, res);
        if (isalpha(s[i])) {
            // toggle case
            s[i] ^= (1 << 5);//大小写转换
            backtrack(s, i + 1, res);
        }
    }

    vector<string> letterCasePermutation(string S) {
        vector<string> res;
        backtrack(S, 0, res);
        return res;
    }

/*
788. 旋转数字

我们称一个数 X 为好数, 如果它的每位数字逐个地被旋转 180 度后，我们仍可以得到一个有效的，且和 X 不同的数。
要求每位数字都要被旋转。如果一个数的每位数字被旋转以后仍然还是一个数字， 则这个数是有效的。0, 1, 
和 8 被旋转后仍然是它们自己；2 和 5 可以互相旋转成对方；6 和 9 同理，除了这些以外其他的数字旋转以后都不再是有效的数字。
现在我们有一个正整数 N, 计算从 1 到 N 中有多少个数 X 是好数？

示例:
输入: 10
输出: 4
解释: 
在[1, 10]中有四个好数： 2, 5, 6, 9。
注意 1 和 10 不是好数, 因为他们在旋转之后不变。
*/
int rotatedDigits(int N) {
        int counts = 0;  //好数的个数

        for(int i = 1;i<=N;i++)

        {

            string temp = "";

            string a = to_string(i);

            for(int j = 0;j<(int)a.size();j++)

            {

                if(a[j]=='3' || a[j]=='4' || a[j]=='7')

                    break;

                else

                {

                    if(a[j]=='0' || a[j]=='1' || a[j]=='8')

                        temp += a[j];

                    else if(a[j]=='2')

                        temp += '5';

                    else if(a[j]=='5')

                        temp += '2';

                    else if(a[j]=='6')

                        temp += '9';

                    else

                        temp += '6';

                }

            }

            if(temp.size()==a.size() && a!=temp)

                counts++;

        }

        return counts;
}

/*
796. 旋转字符串

给定两个字符串, A 和 B。
A 的旋转操作就是将 A 最左边的字符移动到最右边。 例如, 若 A = 'abcde'，在移动一次之后结果就是'bcdea' 。
如果在若干次旋转操作之后，A 能变成B，那么返回True。

示例 1:
输入: A = 'abcde', B = 'cdeab'
输出: true
*/
bool rotateString(string A, string B) {
	return A.size()==B.size() && (A+A).find(B)!=(A+A).npos;
}

/*
804. 唯一摩尔斯密码词

国际摩尔斯密码定义一种标准编码方式，将每个字母对应于一个由一系列点和短线组成的字符串， 
比如: "a" 对应 ".-", "b" 对应 "-...", "c" 对应 "-.-.", 等等。
为了方便，所有26个英文字母对应摩尔斯密码表如下：
[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
给定一个单词列表，每个单词可以写成每个字母对应摩尔斯密码的组合。例如，"cab" 可以写成 "-.-..--..."，
(即 "-.-." + "-..." + ".-"字符串的结合)。我们将这样一个连接过程称作单词翻译。
返回我们可以获得所有词不同单词翻译的数量。

例如:
输入: words = ["gin", "zen", "gig", "msg"]
输出: 2
解释: 
各单词翻译如下:
"gin" -> "--...-."
"zen" -> "--...-."
"gig" -> "--...--."
"msg" -> "--...--."
共有 2 种不同翻译, "--...-." 和 "--...--.".
*/
int uniqueMorseRepresentations(vector<string>& words) {
	        string code [] = {".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.",
            "---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."};
        vector <string> v = words;
        set <string> s;
        for(unsigned int j = 0;j< v.size();j++) {   //遍历words容器
            string temp;   //用来拼接字符串
            for(unsigned i = 0;i<words[j].length();i++) {   //遍历容器中的元素，将其转换为摩尔斯密码
                string word = words[j];
                temp += code[word[i] - 'a'];
        }
                s.insert(temp);   //将拼接好的字符串插入容器
        }
        return s.size();     //输出容器中元素的个数
}

/*
806. 写字符串需要的行数

我们要把给定的字符串 S 从左到右写到每一行上，每一行的最大宽度为100个单位，
如果我们在写某个字母的时候会使这行超过了100 个单位，那么我们应该把这个字母写到下一行。
我们给定了一个数组 widths ，这个数组 widths[0] 代表 'a' 需要的单位，
 widths[1] 代表 'b' 需要的单位，...， widths[25] 代表 'z' 需要的单位。
现在回答两个问题：至少多少行能放下S，以及最后一行使用的宽度是多少个单位？将你的答案作为长度为2的整数列表返回。

示例 1:
输入: 
widths = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
S = "abcdefghijklmnopqrstuvwxyz"
输出: [3, 60]
解释: 
所有的字符拥有相同的占用单位10。所以书写所有的26个字母，
我们需要2个整行和占用60个单位的一行。
*/
vector<int> numberOfLines(vector<int>& widths, string S) {
	int curLen=0,line=1;
	for(int i=0;i<(int)S.size();++i){
		curLen+=widths[S[i]-'a'];

		if(curLen>100){
			++line;
			curLen=widths[S[i]-'a'];
		}
	}

	std::vector<int> v;
	v.push_back(line);
	v.push_back(curLen);

	return v;
}

/*
811. 子域名访问计数

一个网站域名，如"discuss.leetcode.com"，包含了多个子域名。
作为顶级域名，常用的有"com"，下一级则有"leetcode.com"，最低的一级为"discuss.leetcode.com"。
当我们访问域名"discuss.leetcode.com"时，也同时访问了其父域名"leetcode.com"以及顶级域名 "com"。
给定一个带访问次数和域名的组合，要求分别计算每个域名被访问的次数。其格式为访问次数+空格+地址，
例如："9001 discuss.leetcode.com"。接下来会给出一组访问次数和域名组合的列表cpdomains 。
要求解析出所有域名的访问次数，输出格式和输入格式相同，不限定先后顺序。

示例 1:
输入: 
["9001 discuss.leetcode.com"]
输出: 
["9001 discuss.leetcode.com", "9001 leetcode.com", "9001 com"]
说明: 
例子中仅包含一个网站域名："discuss.leetcode.com"。按照前文假设，子域名"leetcode.com"和"com"都会被访问，所以它们都被访问了9001次。
*/
vector<string> subdomainVisits(vector<string>& cpdomains) {
      map<string,int> domain;
        
        vector<string> output;    
        
        for(int i=0; i<(int)cpdomains.size(); i++){
            
            size_t pos1 = cpdomains[i].find(" ");//找出空格的位置
            string temp = cpdomains[i].substr(0, pos1);//截取出表示访问次数的字符串
            
            int count = atoi(temp.c_str());//转为int
            
            if(pos1!=cpdomains[i].npos){

                string temp = cpdomains[i].substr(pos1+1,cpdomains[i].size()-1);//截取出最低一级的域名

                //output.push_back(temp);
                domain[temp] += count;//向map中插入最低一级的域名
                //cout << temp << endl;
            }
            
            
            size_t pos2 = cpdomains[i].find(".");//找出"."的位置
            
            while(pos2!=cpdomains[i].npos){
                
                string temp = cpdomains[i].substr(pos2+1,cpdomains[i].size()-1);//依次截取高级域名
                domain[temp] += count;//向map中插入域名
                
                pos2 = cpdomains[i].find(".", pos2+1);//查找下一个"."
                //cout << temp << endl;
            }
        }
        
        map<string,int>::iterator it;
        
        for(it = domain.begin(); it!=domain.end(); it++){
            
            string temp = std::to_string(it->second) + " " + it->first;//注意将int转为string
            output.push_back(temp);
            
            //cout << it->first << ":" << it->second << endl;
        }
        
        //cout << domain.size() <<endl;

        
        return output;
}

/*
812. 最大三角形面积

给定包含多个点的集合，从其中取三个点组成三角形，返回能组成的最大三角形的面积。

示例:
输入: points = [[0,0],[0,1],[1,0],[0,2],[2,0]]
输出: 2
解释: 
这五个点如下图所示。组成的橙色三角形是最大的，面积为2。
*/
double largestTriangleArea(vector<vector<int>>& points) {
	        double result = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        for (int i = 0; i < (int)points.size() - 2; i++) {
            for (int j = i + 1; j < (int)points.size() - 1; j++) {
                x1 = points[i][0] - points[j][0];
                y1 = points[i][1] - points[j][1];
                for (int k = j + 1; k < (int)points.size(); k++) {
                    x2 = points[i][0] - points[k][0];
                    y2 = points[i][1] - points[k][1];
                    result = max(result, fabs(x1 * y2 - x2 * y1));
                }
            }
        }
        return result / 2.0;
}

/*
819. 最常见的单词

给定一个段落 (paragraph) 和一个禁用单词列表 (banned)。返回出现次数最多，同时不在禁用列表中的单词。
题目保证至少有一个词不在禁用列表中，而且答案唯一。
禁用列表中的单词用小写字母表示，不含标点符号。段落中的单词不区分大小写。答案都是小写字母。

示例：
输入: 
paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]
输出: "ball"
解释: 
"hit" 出现了3次，但它是一个禁用的单词。
"ball" 出现了2次 (同时没有其他单词出现2次)，所以它是段落里出现次数最多的，且不在禁用列表中的单词。 
注意，所有这些单词在段落里不区分大小写，标点符号需要忽略（即使是紧挨着单词也忽略， 比如 "ball,"）， 
"hit"不是最终的答案，虽然它出现次数更多，但它在禁用单词列表中。
*/
string mostCommonWord(string paragraph, vector<string>& banned) {
	     unordered_map<string, int> count;
        unordered_set<string> s;
        string result;
        int c = -1;
        for (auto ban : banned) s.insert(ban);
        string temp;
        for (int i = 0; i < (int)paragraph.size(); i++)
        {
            if ((paragraph[i] >= 'a' && paragraph[i] <= 'z') || (paragraph[i] >= 'A' && paragraph[i] <= 'Z'))
            {
                if (paragraph[i] >= 'a') temp += paragraph[i];
                else temp += paragraph[i] - 'A' + 'a';
            }
            else 
            {
                if (!temp.empty() && s.find(temp) == s.end()) count[temp]++;
                temp.clear();
            }
        }
        if (!temp.empty() && s.find(temp) == s.end()) count[temp]++;
        for (auto item : count)
        {
            if (item.second > c) 
            {
                c = item.second;
                result = item.first;
            }
        }
        return result;
}

/*
821. 字符的最短距离

给定一个字符串 S 和一个字符 C。返回一个代表字符串 S 中每个字符到字符串 S 中的字符 C 的最短距离的数组。

示例 1:
输入: S = "loveleetcode", C = 'e'
输出: [3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]
*/
vector<int> shortestToChar(string S, char C) {
	std::vector<int> v;
	int pre=-1,last=S.find(C);
	for(int i=0;i<(int)S.size();++i){
		if(S[i]==C){
			v.push_back(0);
			last=i;
		}else{
			if(i<last){
				if(pre==-1){
					v.push_back(last-i);
				}else{
					v.push_back(min(i-pre,last-i));
				}
				
			}else{
				if((int)S.find(C,i)!=last){
					pre=last;
				}

				last=S.find(C,i);
				if(last!=(int)S.npos){
					v.push_back(min(i-pre,last-i));
				}else{
					v.push_back(i-pre);
				}
				
			}
		}
	}

	return v;
}

/*
824. 山羊拉丁文

给定一个由空格分割单词的句子 S。每个单词只包含大写或小写字母。
我们要将句子转换为 “Goat Latin”（一种类似于 猪拉丁文 - Pig Latin 的虚构语言）。
山羊拉丁文的规则如下：
	如果单词以元音开头（a, e, i, o, u），在单词后添加"ma"。
	例如，单词"apple"变为"applema"。
	如果单词以辅音字母开头（即非元音字母），移除第一个字符并将它放到末尾，之后再添加"ma"。
	例如，单词"goat"变为"oatgma"。
	根据单词在句子中的索引，在单词最后添加与索引相同数量的字母'a'，索引从1开始。
	例如，在第一个单词后添加"a"，在第二个单词后添加"aa"，以此类推。
返回将 S 转换为山羊拉丁文后的句子。

示例 1:
输入: "I speak Goat Latin"
输出: "Imaa peaksmaaa oatGmaaaa atinLmaaaaa"
*/
string toGoatLatin(string S) {
	    // unordered_set<char> vowels={'a','A','e','E','i','I','o','O','u','U'};
     //    istringstream ss(S);
        string temp,ans;
     //    string str="ma";
     //    while(ss>>temp){    
     //        str+="a";
     //        if(vowels.find(temp[0])!=vowels.end())     //单词首字母是元音
     //            ans+=temp+str+" ";
     //        else ans+=temp.substr(1)+temp[0]+str+" ";       //单词首字母不是元音
     //    }
     //    ans.pop_back();
        return ans;
}

/*
830. 较大分组的位置

在一个由小写字母构成的字符串 S 中，包含由一些连续的相同字符所构成的分组。
例如，在字符串 S = "abbxxxxzyy" 中，就含有 "a", "bb", "xxxx", "z" 和 "yy" 这样的一些分组。
我们称所有包含大于或等于三个连续字符的分组为较大分组。找到每一个较大分组的起始和终止位置。
最终结果按照字典顺序输出。

示例 1:
输入: "abbxxxxzzy"
输出: [[3,6]]
解释: "xxxx" 是一个起始于 3 且终止于 6 的较大分组。
*/
vector<vector<int>> largeGroupPositions(string S) {
	vector<vector<int>> vec;
	int start=0,iCount=0;
	bool flag=false;
	for(int i=0;i<(int)S.size();++i){

		if(!flag && S[i]==S[i+1]){
			start=i;
			flag=true;
			iCount++;
		}
		else if(flag && S[i]==S[i+1]){
			iCount++;
		}

		else if(flag && (i+1>=(int)S.size()|| S[i]!=S[i+1])){
			iCount++;
			
			if(iCount>=3){
				std::vector<int> v;
				v.push_back(start);
				v.push_back(i);
				vec.push_back(v);
			}
			
			flag=false;
			iCount=0;
		}	
	}

	return vec;
}

/*
832. 翻转图像

给定一个二进制矩阵 A，我们想先水平翻转图像，然后反转图像并返回结果。
水平翻转图片就是将图片的每一行都进行翻转，即逆序。例如，水平翻转 [1, 1, 0] 的结果是 [0, 1, 1]。
反转图片的意思是图片中的 0 全部被 1 替换， 1 全部被 0 替换。例如，反转 [0, 1, 1] 的结果是 [1, 0, 0]。

示例 1:
输入: [[1,1,0],[1,0,1],[0,0,0]]
输出: [[1,0,0],[0,1,0],[1,1,1]]
解释: 首先翻转每一行: [[0,1,1],[1,0,1],[0,0,0]]；
     然后反转图片: [[1,0,0],[0,1,0],[1,1,1]]
*/
vector<vector<int>> flipAndInvertImage(vector<vector<int>>& A) {
	    if (A.empty() || A[0].empty()) return A;
        int R = A.size();
        int C = A[0].size();
        for (int i = 0; i < R; ++i) {
            int l = 0;
            int r = C - 1;
            while (l < r) {
                swap(A[i][l], A[i][r]);
                A[i][l++] ^= 1;
                A[i][r--] ^= 1;
            }
            if (C & 1) A[i][C >> 1] ^= 1;
        }
        return A;
}

/*
836. 矩形重叠

矩形以列表 [x1, y1, x2, y2] 的形式表示，其中 (x1, y1) 为左下角的坐标，(x2, y2) 是右上角的坐标。
如果相交的面积为正，则称两矩形重叠。需要明确的是，只在角或边接触的两个矩形不构成重叠。
给出两个矩形，判断它们是否重叠并返回结果。

示例 1：
输入：rec1 = [0,0,2,2], rec2 = [1,1,3,3]
输出：true
*/
bool isRectangleOverlap(vector<int>& rec1, vector<int>& rec2) {
	int left_x=max(rec1[0],rec2[0]);
	int left_y=max(rec1[1],rec2[1]);
	int right_x=min(rec1[2],rec2[2]);
	int right_y=min(rec1[3],rec2[3]);

	if(right_x > left_x && right_y>left_y){
		return true;
	}
	return false;
}

/*
840. 矩阵中的幻方

3 x 3 的幻方是一个填充有从 1 到 9 的不同数字的 3 x 3 矩阵，其中每行，每列以及两条对角线上的各数之和都相等。
给定一个由整数组成的 grid，其中有多少个 3 × 3 的 “幻方” 子矩阵？（每个子矩阵都是连续的）。

示例：
输入: [[4,3,8,4],
      [9,5,1,9],
      [2,7,6,2]]
输出: 1
解释: 
下面的子矩阵是一个 3 x 3 的幻方：
438
951
276
而这一个不是：
384
519
762
总的来说，在本示例所给定的矩阵中只有一个 3 x 3 的幻方子矩阵。
*/
    bool valid(const vector<vector<int>>& grid, int r, int c) {
        if (grid[r + 1][c + 1] != 5) return false;
        unordered_set<int> s;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                int t = grid[r + i][c + j];
                if (t < 1 || t > 9 || s.count(t)) return false;
                s.insert(t);
            }
        }
        for (int i = 0; i < 3; ++i) {
            int s1 = 0;
            int s2 = 0;
            for (int j = 0; j < 3; ++j) {
                s1 += grid[r + i][c + j];
                s2 += grid[r + j][c + i];
            }
            if (s1 != 15 || s2 != 15) return false;
        }
        // 以上判断已经足够断定幻方是否存在
        return true;
    }
    int numMagicSquaresInside(vector<vector<int>>& grid) {
        if (grid.empty()) return 0;
        int R = grid.size();
        int C = grid[0].size();
        int res = 0;
        for (int i = 0; i < R - 2; ++i) {
            for (int j = 0; j < C - 2; ++j) {
                res += valid(grid, i, j);
            }
        }
        return res;
    }

/*
844. 比较含退格的字符串

给定 S 和 T 两个字符串，当它们分别被输入到空白的文本编辑器后，判断二者是否相等，并返回结果。 # 代表退格字符。

示例 1：
输入：S = "ab#c", T = "ad#c"
输出：true
解释：S 和 T 都会变成 “ac”。
*/
bool backspaceCompare(string S, string T) {
	int i=(int)S.size()-1;
	int j=(int)T.size()-1;

	while(i>=0 || j>=0){
		if(i>=0 && S[i]=='#'){
			int iCount=1;
			--i;
			while(i>=0&&iCount>0){
				if(S[i]=='#'){
					++iCount;
				}else{
					--iCount;
				}
				--i;
			}
		}else if(j>=0 && T[j]=='#'){
			int iCount=1;
			--j;
			while(j>=0&&iCount>0){
				if(T[j]=='#'){
					++iCount;
				}else{
					--iCount;
				}
				--j;
			}
		}else if(i>=0 && j>=0 && S[i]==T[j]){
			--i;--j;
		}else{
			return false;
		}
	}

	if(i==-1 && j==-1){
		return true;
	}

	return false;
}

/*
849. 到最近的人的最大距离

在一排座位（ seats）中，1 代表有人坐在座位上，0 代表座位上是空的。
至少有一个空座位，且至少有一人坐在座位上。
亚历克斯希望坐在一个能够使他与离他最近的人之间的距离达到最大化的座位上。
返回他到离他最近的人的最大距离。

示例 1：
输入：[1,0,0,0,1,0,1]
输出：2
解释：
如果亚历克斯坐在第二个空位（seats[2]）上，他到离他最近的人的距离为 2 。
如果亚历克斯坐在其它任何一个空位上，他到离他最近的人的距离为 1 。
因此，他到离他最近的人的最大距离是 2 。 
*/
int maxDistToClosest(vector<int>& s) {
	        int mx = 1, cnt = -1, sz= s.size();
        for (int i = 1; i < sz - 1; i++) {
            if (s[i - 1] == 1 && s[i] == 0) {  
                cnt = 1;
            }else if (s[i] == 0 && s[i + 1] == 1)  {
                if (++cnt > mx) {
                    mx = cnt;
                    cnt = 0;
                }
            }else if (s[i] == 0) {
               cnt++;
            }
        }
        int lcnt = 0, rcnt = 0;
        for (int i = 0; s[i] == 0; lcnt++, i++);
        for (int i = sz - 1; s[i] == 0; rcnt++, i--);
        
        return max((mx + 1) >> 1, max(lcnt, rcnt) );
}

/*
852. 山脉数组的峰顶索引

我们把符合下列属性的数组 A 称作山脉：
	A.length >= 3
	存在 0 < i < A.length - 1 使得A[0] < A[1] < ... A[i-1] < A[i] > A[i+1] > ... > A[A.length - 1]
给定一个确定为山脉的数组，返回任何满足 A[0] < A[1] < ... A[i-1] < A[i] > A[i+1] > ... > A[A.length - 1] 的 i 的值。

示例 1：
输入：[0,1,0]
输出：1
*/
int peakIndexInMountainArray(vector<int>& A) {
	int start=0,end=(int)A.size();
	while(start<end){
		int mid=(start+end)/2;
		if(A[mid-1]<A[mid]&& A[mid]>A[mid+1]){
			return mid;
		}else if(A[mid]>A[mid-1]){
			start=mid-1;
		}else{
			end=mid+1;
		}
	}

	return 0;
}

/*
859. 亲密字符串

给定两个由小写字母构成的字符串 A 和 B ，只要我们可以通过交换 A 中的两个字母得到与 B 相等的结果，
就返回 true ；否则返回 false 。

示例 1：
输入： A = "ab", B = "ba"
输出： true
*/
bool buddyStrings(string A, string B) {
        if(A.empty() || B.empty()){
        return false;
    }
    if(A.size() != B.size()){
        return false;
    }
    vector<int> index;
    set<char> seen;
    bool res1=false;

    for(int i=0; i<(int)A.size(); i++){
        if(seen.count(A[i])){
            res1 = true;
        }
        else{
            seen.insert(A[i]);
        }

        if(A[i] != B[i]){
            index.push_back(i);
        }
    }


    if(index.empty()){
        return res1;
    }
    else{
        if(index.size() != 2){
            return false;
        }
        return (A[index[0]] == B[index[1]] && A[index[1]] == B[index[0]]);
    }
}

/*
860. 柠檬水找零

在柠檬水摊上，每一杯柠檬水的售价为 5 美元。
顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。
每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，
也就是说净交易是每位顾客向你支付 5 美元。
注意，一开始你手头没有任何零钱。
如果你能给每位顾客正确找零，返回 true ，否则返回 false 。

示例 1：
输入：[5,5,5,10,20]
输出：true
解释：
前 3 位顾客那里，我们按顺序收取 3 张 5 美元的钞票。
第 4 位顾客那里，我们收取一张 10 美元的钞票，并返还 5 美元。
第 5 位顾客那里，我们找还一张 10 美元的钞票和一张 5 美元的钞票。
由于所有客户都得到了正确的找零，所以我们输出 true。
*/
bool lemonadeChange(vector<int>& bills) {
        if(bills[0]!=5 || bills[1]==20)//自己初始为0，所以第一回合后自己手中的钱应该为5，第二回合应为10
            return false;
        int n=0;//5元的数目
        int m=0;//10元的数目
        for(int i=0;i<(int)bills.size();i++)
        {
            if(bills[i]==5)
            {
                n++;
            }
            else if(bills[i]==10)
            {
                if(n)
                {
                    n--;
                    m++;
                }
                else return false;
            }
            else
            {
                if(m)
                {
                    if(n)
                    {
                        m--;
                        n--;
                        continue;
                    }
                }
                if(n>=3)
                {
                    n-=3;
                }
                else return false;
            }
        }
        return true;
}

/*
867. 转置矩阵

给定一个矩阵 A， 返回 A 的转置矩阵。
矩阵的转置是指将矩阵的主对角线翻转，交换矩阵的行索引与列索引。

示例 1：
输入：[[1,2,3],[4,5,6],[7,8,9]]
输出：[[1,4,7],[2,5,8],[3,6,9]]
*/
vector<vector<int>> transpose(vector<vector<int>>& A) {
	        if(A.empty())
            return A;
        int rows=A.size(),cols=A[0].size();
        vector<int> temp(rows,0);
        vector<vector<int>> res(cols,temp);
        for(int i=0;i<rows;i++)
            for(int j=0;j<cols;j++)
                res[j][i]=A[i][j];
        return res;
}

/*
868. 二进制间距

给定一个正整数 N，找到并返回 N 的二进制表示中两个连续的 1 之间的最长距离。 
如果没有两个连续的 1，返回 0 。

示例 1：
输入：22
输出：2
解释：
22 的二进制是 0b10110 。
在 22 的二进制表示中，有三个 1，组成两对连续的 1 。
第一对连续的 1 中，两个 1 之间的距离为 2 。
第二对连续的 1 中，两个 1 之间的距离为 1 。
答案取两个距离之中最大的，也就是 2 。
*/
 int binaryGap(int N) {
 	bool flag=false;
 	int iCount=0;
 	int max=0;
 	while(N!=0){
 		if(!flag && (N&1)==1){
 			iCount=1;
 			flag=true;
 			N=N>>1;
 		}else if(flag && (N&1)==0){
 			iCount++;
 			N=N>>1;
 		}else if(flag && (N&1)==1){
 			flag=false;
 			if(iCount>max){
 				max=iCount;
 				iCount=0;
 			}
 		}else{
 			N=N>>1;
 		}
	}

 	return max;
 }

 /*
872. 叶子相似的树

请考虑一颗二叉树上所有的叶子，这些叶子的值按从左到右的顺序排列形成一个 叶值序列 。
举个例子，如上图所示，给定一颗叶值序列为 (6, 7, 4, 9, 8) 的树。
如果有两颗二叉树的叶值序列是相同，那么我们就认为它们是 叶相似 的。
如果给定的两个头结点分别为 root1 和 root2 的树是叶相似的，则返回 true；否则返回 false 。
 */
void leafval(std::vector<int> &v,TreeNode *root){
	if(root==NULL){
		return;
	}

	if(root->left==NULL && root->right==NULL){
		v.push_back(root->val);
	}

	leafval(v,root->left);
	leafval(v,root->right);
}

bool leafSimilar(TreeNode* root1, TreeNode* root2) {
	std::vector<int> v1;
	std::vector<int> v2;

	leafval(v1,root1);
	leafval(v2,root2);

	if(v1.size()==v2.size()){
		for(int i=0;i<(int)v1.size();++i){
			if(v1[i]!=v2[i]){
				return false;
			}
		}
		return true;
	}else{
			return false;
	}
}

int main(){
	// cout<<hasAlternatingBits(1431655764)<<endl;

	// vector<int> res=selfDividingNumbers(1,22);

	// string plate="1s3 PSt";
	// vector<string> words={"step", "steps", "stripe", "stepple"};
	// cout<<shortestCompletingWord(plate,words)<<endl;

	// string a="bbbacddceeb";
    // string b="ceebbbbacdd";
    // rotateString(a,b);

	// string s="abaa";
	// char c='b';
    // shortestToChar(s,c);

	// string s="aaa";
	// largeGroupPositions(s);

	// string S="hd#dp#czsp#####";
	// string T="hd#dp#czsp######";
	// cout<<backspaceCompare(S,T)<<endl;

	cout<<binaryGap(22)<<endl;

	return 0;
}