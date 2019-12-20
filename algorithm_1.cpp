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

using namespace std;
using namespace std::tr1;

/*
392、判断子序列

给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
你可以认为 s 和 t 中仅包含英文小写字母。字符串 t 可能会很长（长度 ~= 500,000），而 s 是个短字符串（长度 <=100）。
字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。

示例 1:
s = "abc", t = "ahbgdc"
返回 true.

示例 2:
s = "axc", t = "ahbgdc"
返回 false.

后续挑战 :
如果有大量输入的 S，称作S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？
*/
bool isSubsequence(string s, string t) {
	int len_s=s.size();
	int len_t=t.size();

	int i=0,j=0;
	while(j<len_t && i<len_s){
		if(s[i]==t[j]){
			++i;
			++j;
		}else{
			++j;
		}
	}

	if(i==len_s){
		return true;
	}

	return false;
}

/*
401. 二进制手表

二进制手表顶部有 4 个 LED 代表小时（0-11），底部的 6 个 LED 代表分钟（0-59）。
每个 LED 代表一个 0 或 1，最低位在右侧。

给定一个非负整数 n 代表当前 LED 亮着的数量，返回所有可能的时间。

案例:
输入: n = 1
返回: ["1:00", "2:00", "4:00", "8:00", "0:01", "0:02", "0:04", "0:08", "0:16", "0:32"]
*/
int count(int n){
	int num=0;
	while(n!=0){
		n=n&(n-1);
		++num;
	}

	return num;
}

vector<string> readBinaryWatch(int num) {
	vector<string> v;

	for(int i=0;i<12;++i){
		for(int j=0;j<60;++j){
			if(count(i)+count(j)==num){
				v.push_back(to_string(i)+':'+(j<10?'0'+to_string(j):to_string(j)));
			}
		}
	}

	return v;
}

/*
404. 左叶子之和

计算给定二叉树的所有左叶子之和。

示例：
    3
   / \
  9  20
    /  \
   15   7

在这个二叉树中，有两个左叶子，分别是 9 和 15，所以返回 24
*/
struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

int sumOfLeftLeaves(TreeNode* root){
	int sum=0;

	if(root==NULL){
		return 0;
	}

	if(root->left!=NULL && root->left->left==NULL && root->left->right==NULL){
		sum+=root->left->val;
	}

	return sum+sumOfLeftLeaves(root->right)+sumOfLeftLeaves(root->left);
}

/*
405. 数字转换为十六进制数

给定一个整数，编写一个算法将这个数转换为十六进制数。对于负整数，我们通常使用 补码运算 方法。
注意:
	十六进制中所有字母(a-f)都必须是小写。
	十六进制字符串中不能包含多余的前导零。如果要转化的数为0，那么以单个字符'0'来表示；对于其他情况，十六进制字符串中的第一个字符将不会是0字符。 
	给定的数确保在32位有符号整数范围内。
	不能使用任何由库提供的将数字直接转换或格式化为十六进制的方法。

示例 1：
输入:26 输出:"1a"
示例 2：
输入:-1 输出:"ffffffff"
*/
string toHex(int num) {
	     if(num==0)
        {
            return "0";
        }

        char a[16]={'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};
        vector<char> temp;
        int count=0;
        char cc;
        while(num&&count<8)
        {
            cc=a[num&0xf];
            temp.push_back(cc);
            count++;
            num>>=4;
        }
        reverse(temp.begin(),temp.end());
        string res="";
        for(int i=0;i<count;i++)
        {
           res+=temp[i];
        }
        return res;  
}

/*
409. 最长回文串

给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。
在构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。
注意:
假设字符串的长度不会超过 1010。

示例 1: 
输入:"abccccdd" 输出:7
解释:我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。
*/
int longestPalindrome(string s) {
	vector<int> arr(128,0);
	int len=0;
	int flag=0;
	for(size_t i = 0;i<s.size();++i){
		arr[s[i]]++;
	}

	for(int i=0;i<128;++i){
		if(arr[i]%2==0){
			len+=arr[i];
		}else{
			len+=arr[i]-1;
			flag=1;
		}
	}

	return len+flag;
}

/*
412. Fizz Buzz

写一个程序，输出从 1 到 n 数字的字符串表示。
1. 如果 n 是3的倍数，输出“Fizz”；
2. 如果 n 是5的倍数，输出“Buzz”；
3.如果 n 同时是3和5的倍数，输出 “FizzBuzz”。

例：n = 15,
返回:
[
    "1",
    "2",
    "Fizz",
    "4",
    "Buzz",
    "Fizz",
    "7",
    "8",
    "Fizz",
    "Buzz",
    "11",
    "Fizz",
    "13",
    "14",
    "FizzBuzz"
]
*/
vector<string> fizzBuzz(int n) {
     vector<string> res(n);
     for(int i=1;i<=n;++i){
     	if(i%3==0){
     		res[i-1]+=string("Fizz");
     	}

     	if(i%5==0){
     		res[i-1]+=string("Buzz");
     	}

     	if(res[i-1]==""){
     		res[i-1]+=to_string(i);
     	}
     }   

     return res;
}

/*
414. 第三大的数

给定一个非空数组，返回此数组中第三大的数。如果不存在，则返回数组中最大的数。要求算法时间复杂度必须是O(n)。

示例 1:
输入: [3, 2, 1] 输出: 1
解释: 第三大的数是 1.

示例 2:
输入: [1, 2] 输出: 2
解释: 第三大的数不存在, 所以返回最大的数 2 .
*/
int thirdMax(vector<int>& nums) {
	long long first,second,next;
	const long long MINN=-3147483648;
	first=MINN;
	second=MINN;
	next=MINN;

	for(size_t i=0;i<nums.size();++i){
		if(nums[i]>first){
			next=second;
			second=first;
			first=nums[i];
		}else if(nums[i]<first && nums[i]>second){
			next=second;
			second=nums[i];
		}else if(nums[i]<first && nums[i]<second && nums[i]>next){
			next=nums[i];
		}
	}

	if(next==MINN) return int(first);

	return int(next);
}

/*
415. 字符串相加

给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和。
注意：
	num1 和num2 的长度都小于 5100.
	num1 和num2 都只包含数字 0-9.
	num1 和num2 都不包含任何前导零。
	你不能使用任何內建 BigInteger 库，也不能直接将输入的字符串转换为整数形式。
*/
string addStrings(string num1, string num2) {
	reverse(num1.begin(),num1.end());
	reverse(num2.begin(),num2.end());
	
	if(num1.length()>num2.length()) swap(num1,num2);
	
	string res;
	size_t i=0;
	int c=0,t;
	for(;i<num1.size();++i){
		if((t=num1[i]+num2[i]-'0'-'0'+c)>9){
			c=1;
		}else{
			c=0;
		}

		t%=10;
		res+=to_string(t);
	}

	while(i<num2.size()){
		if((t=num2[i]-'0'+c)>9){
			c=1;
		}else{
			c=0;
		}
		t%=10;
		res+=to_string(t);
		++i;
	}

	if(c==1) res+="1";

	reverse(res.begin(),res.end());
	return res;
}

/*
434. 字符串中的单词数

统计字符串中的单词个数，这里的单词指的是连续的不是空格的字符。
请注意，你可以假定字符串里不包括任何不可打印的字符。

示例:输入: "Hello, my name is John" 输出: 5
*/
int countSegments(string s) {
	if(s.size()==0){
		return 0;
	}

	int num=0,flag=0;
	for(size_t i=0;i<s.size();++i){
		if(!flag && s[i]!=' '){
			flag=1;
		}
		if(flag && (s[i]==' ' || i==s.size()-1)){
			++num;
			flag=0;
		}
	}

	return num;
}

/*
437. 路径总和 III

给定一个二叉树，它的每个结点都存放着一个整数值。找出路径和等于给定数值的路径总数。
路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。
示例：
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

返回 3。和等于 8 的路径有:
1.  5 -> 3
2.  5 -> 2 -> 1
3.  -3 -> 11
*/
// struct TreeNode {
//      int val;
//      TreeNode *left;
//      TreeNode *right;
//      TreeNode(int x) : val(x), left(NULL), right(NULL) {}
// };
int dfs(TreeNode *root,int sum){
	int res=0;

	if(root==NULL){
		return res;
	}

	if(sum==root->val){
		++res;
	}

	res+=dfs(root->left,sum-root->val);
	res+=dfs(root->right,sum-root->val);

	return res;
}

int pathSum(TreeNode* root, int sum) {
	if(root==NULL){
		return 0;
	}

	return dfs(root,sum)+pathSum(root->left,sum)+pathSum(root->right,sum);
}

/*
441. 排列硬币

你总共有 n 枚硬币，你需要将它们摆成一个阶梯形状，第 k 行就必须正好有 k 枚硬币。
给定一个数字 n，找出可形成完整阶梯行的总行数。n 是一个非负整数，并且在32位有符号整型的范围内。

示例 1:n = 5 硬币可排列成以下几行:
¤
¤ ¤
¤ ¤
因为第三行不完整，所以返回2.

示例 2:n = 8 硬币可排列成以下几行:
¤
¤ ¤
¤ ¤ ¤
¤ ¤
因为第四行不完整，所以返回3.
*/
int arrangeCoins(int n) {
	return sqrt(8*n+1)/2;
}

/*
443. 压缩字符串

给定一组字符，使用原地算法将其压缩。压缩后的长度必须始终小于或等于原数组长度。
数组的每个元素应该是长度为1 的字符（不是 int 整数类型）。在完成原地修改输入数组后，返回数组的新长度。

进阶：
你能否仅使用O(1) 空间解决问题？

示例 1：
输入：["a","a","b","b","c","c","c"] 
输出：返回6，输入数组的前6个字符应该是：["a","2","b","2","c","3"]
说明："aa"被"a2"替代。"bb"被"b2"替代。"ccc"被"c3"替代。

示例 2：
输入：["a"]
输出：返回1，输入数组的前1个字符应该是：["a"]
说明：没有任何字符串被替代。


示例 3：
输入：["a","b","b","b","b","b","b","b","b","b","b","b","b"]
输出：返回4，输入数组的前4个字符应该是：["a","b","1","2"]。
说明：由于字符"a"不重复，所以不会被压缩。"bbbbbbbbbbbb"被“b12”替代。注意每个数字在数组中都有它自己的位置。
*/
int compress(vector<char>& chars) {
    int num=1;
    for(size_t i=1;i<chars.size();i++)
    {
	            if(chars[i]==chars[i-1])
	            {
	                num+=1;
	                chars.erase(chars.begin()+i);
	                i--;
	            }
	            else
	            {
	                if(num>1)
	                {
	                    string s=to_string(num);
	                    for(auto cc:s)
	                    {
	                        chars.insert(chars.begin()+i,cc);
	                        i++;
	                    }
	                    num=1;
	         }
        }
    }

    if(num>1)
    {
        string s=to_string(num);
        for(auto cc:s)
        {
            chars.push_back(cc);
        }
    }
    return chars.size();
}

/*
447. 回旋镖的数量

给定平面上 n 对不同的点，“回旋镖” 是由点表示的元组 (i, j, k) ，其中 i 和 j 之间的距离和 i 和 k 之间的距离相等（需要考虑元组的顺序）。
找到所有回旋镖的数量。你可以假设 n 最大为 500，所有点的坐标在闭区间 [-10000, 10000] 中。

示例:
输入:[[0,0],[1,0],[2,0]]
输出:2
解释:两个回旋镖为 [[1,0],[0,0],[2,0]] 和 [[1,0],[2,0],[0,0]]
*/
int numberOfBoomerangs(vector<vector<int>>& points) {
	long len = points.size();        
    if(len == 0 || len == 1)            
          return 0;        
          
    unordered_map<int, int> distance;        
    int ret = 0, x, y;        
    for(int i = 0; i < len; i++){            
        for(int j = 0; j < len; j++){                
            x = (points[i][0]-points[j][0]);
            y = (points[i][1] - points[j][1]);                
            distance[x*x + y*y]++;            
        }            
                  
       	for(auto it : distance) 
       		ret += it.second * (it.second - 1);   
        distance.clear();               
    }        
         
    return ret;
}

/*
448. 找到所有数组中消失的数字

给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
找到所有在 [1, n] 范围之间没有出现在数组中的数字。
您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。

示例:
输入:[4,3,2,7,8,2,3,1]
输出:[5,6]
*/
vector<int> findDisappearedNumbers(vector<int>& nums) {
	std::vector<int> v;

	for(size_t i=0;i<nums.size();++i){
		int val=abs(nums[i])-1;
		if(nums[val]>0) nums[val]=-nums[val];
	}

	for(size_t i=0;i<nums.size();++i){
		if(nums[i]>0){
			v.push_back(i+1);
		}
	}

	return v;
}

/*
453. 最小移动次数使数组元素相等

给定一个长度为 n 的非空整数数组，找到让数组所有元素相等的最小移动次数。每次移动可以使 n - 1 个元素增加 1。

示例:
输入:[1,2,3]
输出:3
解释:只需要3次移动（注意每次移动会增加两个元素的值）：[1,2,3]  =>  [2,3,3]  =>  [3,4,3]  =>  [4,4,4]
*/
int minMoves(vector<int>& nums) {
	int min=*min_element(nums.begin(),nums.end());

	int nCount=0;
	for(size_t i=0;i<nums.size();++i){
		nCount=nCount+nums[i]-min;
	}

	return nCount;
}

/*
455. 分发饼干

假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。对每个孩子 i ，都有一个胃口值 gi ，
这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j ，都有一个尺寸 sj 。如果 sj >= gi ，我们可以将这个饼干 j 分配给孩子 i ，
这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。
注意：你可以假设胃口值为正。一个小朋友最多只能拥有一块饼干。

示例 1:
输入: [1,2,3], [1,1] 输出: 1
解释: 
你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
所以你应该输出1。
*/
int findContentChildren(vector<int>& g, vector<int>& s) {
	sort(g.begin(),g.end());
	sort(s.begin(),s.end());

	size_t i=0,j=0;
	int res=0;
	while(i<g.size() && j<s.size()){
		if(s[j]>=g[i]){
			++i;
			++j;
			++res;
		}else{
			++j;
		}
	}

	return res;
}

/*
459. 重复的子字符串

给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度不超过10000。

示例 1:输入: "abab" 输出: True
解释: 可由子字符串 "ab" 重复两次构成。
*/
bool repeatedSubstringPattern(string s) {
	string ss = s + s;
    string ss_(ss.begin() + 1, ss.end() - 1);
    return (ss_.find(s)<=ss.size());
}


/*
461. 汉明距离

两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。
给出两个整数 x 和 y，计算它们之间的汉明距离。

示例:
输入: x = 1, y = 4
输出: 2
解释:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑
*/
int hammingDistance(int x, int y) {
	int i=x^y;
	int nCount=0;
	while(i){
		++nCount;
		i=(i-1)&i;
	}
	
	return nCount;
}

/*
463. 岛屿的周长

给定一个包含 0 和 1 的二维网格地图，其中 1 表示陆地 0 表示水域。
网格中的格子水平和垂直方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。
岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100 。计算这个岛屿的周长。

示例 :

输入:
[[0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [1,1,0,0]]
输出: 16
解释: 它的周长是下面图片中的 16 个黄色的边：
*/
int islandPerimeter(vector<vector<int>>& grid) {
	size_t height=grid.size();
	size_t width=grid[0].size();

	std::vector<std::vector<int>> vecNewGrid(height+2);
	for(size_t i=0;i<height+2;++i){
		vecNewGrid[i].resize(width+2);
		for(size_t j=0;j<width+2;++j){

			if(i==0 || j==0 || i== height+1 || j == width+1){
				vecNewGrid[i][j]=0;
			}else{
				vecNewGrid[i][j]=grid[i-1][j-1];
			}
		}
	}


	int ans=0;
	for(size_t i=0;i<height+2;++i){
		for(size_t j=0;j<width+2;++j){
			if(vecNewGrid[i][j]==1){
				if(vecNewGrid[i-1][j]==0){
					++ans;
				}
				if(vecNewGrid[i+1][j]==0){
					++ans;
				}
				if(vecNewGrid[i][j-1]==0){
					++ans;
				}
				if(vecNewGrid[i][j+1]==0){
					++ans;
				}
			}
		}
	}

	return ans;
}

/*
475. 供暖器
冬季已经来临。 你的任务是设计一个有固定加热半径的供暖器向所有房屋供暖。
现在，给出位于一条水平线上的房屋和供暖器的位置，找到可以覆盖所有房屋的最小加热半径。
所以，你的输入将会是房屋和供暖器的位置。你将输出供暖器的最小加热半径。
说明:
	给出的房屋和供暖器的数目是非负数且不会超过 25000。
	给出的房屋和供暖器的位置均是非负数且不会超过10^9。
	只要房屋位于供暖器的半径内(包括在边缘上)，它就可以得到供暖。
	所有供暖器都遵循你的半径标准，加热的半径也一样。
示例 1:
输入: [1,2,3],[2]
输出: 1
解释: 仅在位置2上有一个供暖器。如果我们将加热半径设为1，那么所有房屋就都能得到供暖。
*/
int findRadius(vector<int>& houses, vector<int>& heaters) {
	sort(houses.begin(), houses.end());
    sort(heaters.begin(), heaters.end());
    size_t i = 0;
    int res = 0;
    for(auto house : houses) {
        while(i < heaters.size() - 1 && abs(heaters[i] - house) >= abs(heaters[i+1] - house))
        {
            i++;
        }
        res = max(res, abs(heaters[i] - house));
    }

    return res;
}

/*
476. 数字的补数

给定一个正整数，输出它的补数。补数是对该数的二进制表示取反。
注意:
	给定的整数保证在32位带符号整数的范围内。
	你可以假定二进制数不包含前导零位。

示例 1:
输入: 5
输出: 2
解释: 5的二进制表示为101（没有前导零位），其补数为010。所以你需要输出2。
*/
int findComplement(int num) {
	int nCount=0;
	while(pow(2,nCount)-1<num){
		nCount++;
	}

	int nMaxValue=pow(2,nCount)-1;

	return nMaxValue^num;
}

/*
482. 密钥格式化

给定一个密钥字符串S，只包含字母，数字以及 '-'（破折号）。N 个 '-' 将字符串分成了 N+1 组。给定一个数字 K，重新格式化字符串，
除了第一个分组以外，每个分组要包含 K 个字符，第一个分组至少要包含 1 个字符。两个分组之间用 '-'（破折号）隔开，并且将所有的小写字母转换为大写字母。
给定非空字符串 S 和数字 K，按照上面描述的规则进行格式化。

示例 1：
输入：S = "5F3Z-2e-9-w", K = 4
输出："5F3Z-2E9W"
解释：字符串 S 被分成了两个部分，每部分 4 个字符；注意，两个额外的破折号需要删掉。
*/
string licenseKeyFormatting(string S, int K) {
	int count=0;        
	for(int i=S.size()-1;i>=0;i--)        
	{            
		if(S[i]=='-')               
			S.erase(S.begin()+i);            
		else             
		{                
			count++;               
		if(S[i]<='z' && S[i]>='a')                    
			S[i]-=32;                
		if(count%K==0 && i!=0)                    
			if(i!=0)                        
				S.insert(S.begin()+i,'-');            
		}        
	}        
	while(S[0]=='-')            
		S.erase(S.begin());        

	return S;
}

/*
485. 最大连续1的个数

给定一个二进制数组， 计算其中最大连续1的个数。

示例 1:
输入: [1,1,0,1,1,1]
输出: 3
解释: 开头的两位和最后的三位都是连续1，所以最大连续1的个数是 3.
*/
int findMaxConsecutiveOnes(vector<int>& nums) {
	int maxCount=0;
	bool flag=false;
	int nCount=0;
	for(size_t i=0;i<nums.size();++i){
		if(!flag && nums[i]==1){
			flag=true;
			++nCount;
		}
		// 1,0,1,1,0,1
		else if(flag && nums[i]==1){
			++nCount;
		}

		else if(flag && nums[i]!=1){
			if(nCount>maxCount){
				maxCount=nCount;
			}
			cout<<maxCount<<endl;
			nCount=0;
			flag=false;
		}
	}

	if(nCount>maxCount){
		maxCount=nCount;
	}

	return maxCount;
}

/*
492. 构造矩形

作为一位web开发者， 懂得怎样去规划一个页面的尺寸是很重要的。 现给定一个具体的矩形页面面积，
你的任务是设计一个长度为 L 和宽度为 W 且满足以下要求的矩形的页面。要求：
1. 你设计的矩形页面必须等于给定的目标面积。
2. 宽度 W 不应大于长度 L，换言之，要求 L >= W 。
3. 长度 L 和宽度 W 之间的差距应当尽可能小。
你需要按顺序输出你设计的页面的长度 L 和宽度 W。

示例：
输入: 4
输出: [2, 2]
解释: 目标面积是 4， 所有可能的构造方案有 [1,4], [2,2], [4,1]。
但是根据要求2，[1,4] 不符合要求; 根据要求3，[2,2] 比 [4,1] 更能符合要求. 所以输出长度 L 为 2， 宽度 W 为 2。
*/
 vector<int> constructRectangle(int area) {
 	int n=sqrt(area);

 	int minDistance=area;
 	int l=0,w=0;
 	for(int i=1;i<=n;++i){
 		int j=area/i;
 		if(i*j==area && j-i <minDistance){
 			minDistance=j-i;
 			l=j;
 			w=i;
 		}
 	}

 	std::vector<int> v={l,w};
 	return v;
 }

/*
496. 下一个更大元素 I

给定两个没有重复元素的数组 nums1 和 nums2 ，其中nums1 是 nums2 的子集。
找到 nums1 中每个元素在 nums2 中的下一个比其大的值。
nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出-1。

示例 1:
输入: nums1 = [4,1,2], nums2 = [1,3,4,2].
输出: [-1,3,-1]
解释:对于num1中的数字4，你无法在第二个数组中找到下一个更大的数字，因此输出 -1。
对于num1中的数字1，第二个数组中数字1右边的下一个较大数字是 3。
对于num1中的数字2，第二个数组中没有下一个更大的数字，因此输出 -1。
*/
vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
	size_t len2=nums2.size();
	std::vector<int> vecNextIndex;

	for(size_t i=0;i<nums1.size();++i){
		size_t j=0,pos=len2;
		while(j<len2){
			if(nums2[j]==nums1[i]){
				pos=j;
			}

			if(nums2[j]>nums1[i] && j>pos){
				vecNextIndex.push_back(nums2[j]);
				break;
			}

			++j;
		}

		if(j==len2){
			vecNextIndex.push_back(-1);
		}
	}

	return vecNextIndex;
}

/*
500. 键盘行

给定一个单词列表，只返回可以使用在键盘同一行的字母打印出来的单词。键盘如下图所示。

示例：
输入: ["Hello", "Alaska", "Dad", "Peace"]
输出: ["Alaska", "Dad"]
*/
vector<string> findWords(vector<string>& words) {
	string L1 = "QWERTYUIOP";//"QWERTYUIOPqwertyuiop";
        string L2 = "ASDFGHJKL";//"ASDFGHJKLasdfghjkl";
        string L3 = "ZXCVBNM";//"ZXCVBNMzxcvbnm";
		vector<string> Fwords;
        for (size_t i = 0; i < words.size(); i++)
        {
            string strTemp(words[i]);
            size_t sum1 = 0, sum2 = 0, sum3 = 0;
            transform(words[i].begin(), words[i].end(), strTemp.begin(), ::toupper);
            for (size_t j = 0; j < strTemp.size(); j++)
            {
                if (L1.find(strTemp[j]) != L1.npos)sum1++;
                if (L2.find(strTemp[j]) != L2.npos)sum2++;
                if (L3.find(strTemp[j]) != L3.npos)sum3++;
            }
            if (sum1 == strTemp.size()) Fwords.push_back(words[i]);
            if (sum2 == strTemp.size()) Fwords.push_back(words[i]);
            if (sum3 == strTemp.size()) Fwords.push_back(words[i]);
        }
	 	return Fwords;
}

/*
501. 二叉搜索树中的众数

给定一个有相同值的二叉搜索树（BST），找出 BST 中的所有众数（出现频率最高的元素）。
假定 BST 有如下定义：
	结点左子树中所含结点的值小于等于当前结点的值
	结点右子树中所含结点的值大于等于当前结点的值
	左子树和右子树都是二叉搜索树

例如：
给定 BST [1,null,2,2],
   1
    \
     2
    /
   2
返回[2].

提示：如果众数超过1个，不需考虑输出顺序
进阶：你可以不使用额外的空间吗？（假设由递归产生的隐式调用栈的开销不被计算在内）
*/
/* struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
vector<int> findMode_res;
int findMode_mx = 0, findMode_cnt = 1;

 void inorder(TreeNode* node, TreeNode*& pre) {
        if(!node) {
            return ;
        }
        inorder(node->left, pre);
        if(pre) {
            findMode_cnt = (node->val == pre->val) ? findMode_cnt + 1 : 1;
        }
        if(findMode_cnt >= findMode_mx) {
            if(findMode_cnt > findMode_mx) {
                findMode_res.clear();
            }
            findMode_res.push_back(node->val);
            findMode_mx = findMode_cnt;
        }
        pre = node;
        inorder(node->right, pre);
    }

vector<int> findMode(TreeNode* root) {
	TreeNode* pre = NULL;
    inorder(root, pre);
    return findMode_res;
}

/*
504. 七进制数

给定一个整数，将其转化为7进制，并以字符串形式输出。

示例 1:
输入: 100输出: "202"
示例 2:
输入: -7 输出: "-10"
*/
string convertToBase7(int num) {
	if(num<7 && num>-7){
		return to_string(num);
	}

	int m=num>0?num%7:-num%7;
	int n=num/7;
	return convertToBase7(n)+to_string(m);
}

/*
506. 相对名次

给出 N 名运动员的成绩，找出他们的相对名次并授予前三名对应的奖牌。前三名运动员将会被分别授予 “金牌”，
“银牌” 和“ 铜牌”（"Gold Medal", "Silver Medal", "Bronze Medal"）。
(注：分数越高的选手，排名越靠前。)

示例 1:

输入: [5, 4, 3, 2, 1]
输出: ["Gold Medal", "Silver Medal", "Bronze Medal", "4", "5"]
解释: 前三名运动员的成绩为前三高的，因此将会分别被授予 “金牌”，“银牌”和“铜牌” ("Gold Medal", "Silver Medal" and "Bronze Medal").
余下的两名运动员，我们只需要通过他们的成绩计算将其相对名次即可。
*/
vector<string> findRelativeRanks(vector<int>& nums) {
	    int len=nums.size();
        vector<string> v(nums.size());
        map<int,int> m;
        for(size_t i=0;i<nums.size();i++){
            m[nums[i]]=i;
        }
        for(auto j:m){
            if(len==3){ v[j.second]="Bronze Medal";len--;}
            else if(len==2){v[j.second]="Silver Medal";len--;}
            else if(len==1){v[j.second]="Gold Medal";}
            else
            v[j.second]=to_string(len--);
        }
        return v;
}

/*
507. 完美数

对于一个 正整数，如果它和除了它自身以外的所有正因子之和相等，我们称它为“完美数”。
给定一个 整数 n， 如果他是完美数，返回 True，否则返回 False

示例：
输入: 28 输出: True
解释: 28 = 1 + 2 + 4 + 7 + 14
*/
bool checkPerfectNumber(int num) {
	if(num==1){
		return false;
	}

	int cnt=1;
	for(int i=2;i<sqrt(num);++i){
		if(num%i==0){
			cnt+=i;
			cnt+=num/i;
		}
	}

	return cnt==num;
}

/*
509. 斐波那契数

斐波那契数，通常用 F(n) 表示，形成的序列称为斐波那契数列。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
给定 N，计算 F(N)。

示例 1：
输入：2 输出：1
解释：F(2) = F(1) + F(0) = 1 + 0 = 1.
*/
int fib(int N) {
	if(N==0){
		return 0;
	}

	if(N==1){
		return 1;
	}

	int f1=1,f0=0;
	int res=0;
	for(int i=2;i<=N;++i){
		res=f1+f0;
		f0=f1;
		f1=res;
	}

	return res;
}

/*
520. 检测大写字母

给定一个单词，你需要判断单词的大写使用是否正确。
我们定义，在以下情况时，单词的大写用法是正确的：
	全部字母都是大写，比如"USA"。
	单词中所有字母都不是大写，比如"leetcode"。
	如果单词不只含有一个字母，只有首字母大写， 比如 "Google"。
否则，我们定义这个单词没有正确使用大写字母。

示例 1:
输入: "USA" 输出: True
*/
bool detectCapitalUse(string word) {
       if(word.empty())
           return false;
       int len  = word.length();
       int upperNum = 0;
       for(int i = 0; i < len; i++)
       {
           if(isupper(word[i]))
               upperNum++;
       }
       if(len == upperNum || 0 == upperNum)
           return true;
       if(1 == upperNum && isupper(word[0]))
           return true;
       return false;
}

/*
521. 最长特殊序列 Ⅰ

给定两个字符串，你需要从这两个字符串中找出最长的特殊序列。最长特殊序列定义如下：该序列为某字符串独有的最长子序列（即不能是其他字符串的子序列）。
子序列可以通过删去字符串中的某些字符实现，但不能改变剩余字符的相对顺序。空序列为所有字符串的子序列，任何字符串为其自身的子序列。
输入为两个字符串，输出最长特殊序列的长度。如果不存在，则返回 -1。

示例 :

输入: "aba", "cdc"
输出: 3
解析: 最长特殊序列可为 "aba" (或 "cdc")
*/
int findLUSlength(string a, string b) {
	int alen=a.size();
    int blen=b.size();
    if(alen!=blen)
    {
        return max(alen,blen);
    }
    else
    {
        if(a==b) return -1;
        else return alen;
    }
}

/*
530. 二叉搜索树的最小绝对差

给定一个所有节点为非负值的二叉搜索树，求树中任意两节点的差的绝对值的最小值。

示例 :
输入:
   1
    \
     3
    /
   2
输出:1
解释:最小绝对差为1，其中 2 和 1 的差的绝对值为 1（或者 2 和 3）。
[543,384,652,null,445,null,699]
*/
 /* 
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
void getMinimumDifference_inorder(TreeNode *root,std::vector<int> &v){
	if(root==NULL){
		return ;
	}

	getMinimumDifference_inorder(root->left,v);
	v.push_back(root->val);
	getMinimumDifference_inorder(root->right,v);
}

int getMinimumDifference(TreeNode* root) {
	std::vector<int> v;
	getMinimumDifference_inorder(root,v);
	int res=9999999;
	sort(v.begin(),v.end());
	for(size_t i=1;i<v.size();++i){
		 res=min(res,v[i]-v[i-1]);
	}

	return res;
}

/*
532. 数组中的K-diff数对

给定一个整数数组和一个整数 k, 你需要在数组里找到不同的 k-diff 数对。
这里将 k-diff 数对定义为一个整数对 (i, j), 其中 i 和 j 都是数组中的数字，且两数之差的绝对值是 k.

示例 1:
输入: [3, 1, 4, 1, 5], k = 2 输出: 2
解释: 数组中有两个 2-diff 数对, (1, 3) 和 (3, 5)。
尽管数组中有两个1，但我们只应返回不同的数对的数量。
*/
int findPairs(vector<int>& nums, int k) {
	int res=0;
	unordered_map<int,int> mp;
	for(auto x:nums) mp[x]++;

	for(auto a:mp){
		if(k==0 && a.second>1) ++res;
		if(k>0 && mp.count(a.first+k)) ++res;
	}

	return res;
}

/*
538. 把二叉搜索树转换为累加树

给定一个二叉搜索树（Binary Search Tree），把它转换成为累加树（Greater Tree)，
使得每个节点的值是原来的节点值加上所有大于它的节点值之和。

例如：
输入: 二叉搜索树:
              5
            /   \
           2     13
输出: 转换为累加树:
             18
            /   \
          20     13
*/
void convertBST_inorder(TreeNode *root,int &pred){
	if(root==NULL) return ;

	convertBST_inorder(root->right,pred);

	pred+=root->val;
	root->val=pred;

	convertBST_inorder(root->left,pred);
}

TreeNode* convertBST(TreeNode* root) {
	int pred=0;
	convertBST_inorder(root,pred);

	return root;
}

/*
541. 反转字符串 II

给定一个字符串和一个整数 k，你需要对从字符串开头算起的每个 2k 个字符的前k个字符进行反转。
如果剩余少于 k 个字符，则将剩余的所有全部反转。如果有小于 2k 但大于或等于 k 个字符，
则反转前 k 个字符，并将剩余的字符保持原样。

示例:
输入: s = "abcdefg", k = 2
输出: "bacdfeg"
*/
void reversestr(string &s,int start,int end){
	while(start<end){
		char tmp=s[start];
		s[start]=s[end];
		s[end]=tmp;

		++start;
		--end;
	}
}

string reverseStr(string s, int k) {
	int p=0;
	int len=s.size();

	while(p<len){
		if(p<len && p+k > len){
			reversestr(s,p,len-1);
		}else{
			reversestr(s,p,p+k-1);
		}

		p=p+2*k;
	}

	return s;
}

/*
543. 二叉树的直径

给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过根结点。

示例 :
给定二叉树

          1
         / \
        2   3
       / \     
      4   5    
返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。
*/
int get_Max(TreeNode *root,int &res){
	if(root==NULL){
		return 0;
	}

	int left=get_Max(root->left,res);
	int right=get_Max(root->right,res);

	res=max(res,left+right+1);

	return max(right,left)+1;
}

int diameterOfBinaryTree(TreeNode* root) {
	if(root==NULL){
		return 0;
	}

	int res=0;
	get_Max(root,res);

	return res-1;
}

/*
551. 学生出勤记录 I

给定一个字符串来代表一个学生的出勤记录，这个记录仅包含以下三个字符：
	'A' : Absent，缺勤
	'L' : Late，迟到
	'P' : Present，到场
如果一个学生的出勤记录中不超过一个'A'(缺勤)并且不超过两个连续的'L'(迟到),那么这个学生会被奖赏。
你需要根据这个学生的出勤记录判断他是否会被奖赏。

示例 1:
输入: "PPALLP"
输出: True
*/
bool checkRecord(string s) {
	int aCount=0;
	bool flag=false;
	int lCount=0;

	for(size_t i=0;i<s.size();++i){
		if(s[i]=='A'){
			++aCount;
		}

		if(s[i]=='L' && !flag){
			++lCount;
			flag=false;
		}

		if(flag && s[i]=='L'){
			++lCount;
		}

		if(s[i]!='L'){
			flag=false;

			if(lCount<=2){
				lCount=0;
			}
		}
	}

	if(aCount<=1 && lCount<=2){
		return true;
	}


	return false;
}

/*
557. 反转字符串中的单词 III

给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。

示例 1:
输入: "Let's take LeetCode contest"
输出: "s'teL ekat edoCteeL tsetnoc" 
注意：在字符串中，每个单词由单个空格分隔，并且字符串中不会有任何额外的空格。
*/
void swap_words(string &s,int start,int end){
	int p=start,q=end-1;
	while(p<q){
		swap(s[p],s[q]);
		p++;
		q--;
	}
}

string reverseWords(string s) {
	size_t p=0,q=0;
	for( ;q<=s.size();++q){
		if(s[q]==' ' or q==s.size()){
			swap_words(s,p,q);
			p=q+1;
		}
	}

	return s;
}

/*
559. N叉树的最大深度

给定一个 N 叉树，找到其最大深度。
最大深度是指从根节点到最远叶子节点的最长路径上的节点总数。
*/
class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};

int maxDepth(Node* root) {
	if(root==NULL){
		return 0;
	}

	int max=1;
	for(size_t i=0;i<root->children.size();++i){
		int depth=maxDepth(root->children[i])+1;
		if(max<depth){
			max=depth;
		}
	}

	return max;
}

/*
561. 数组拆分 I

给定长度为 2n 的数组, 你的任务是将这些数分成 n 对, 例如 (a1, b1), (a2, b2), ..., (an, bn) ，
使得从1 到 n 的 min(ai, bi) 总和最大。

示例 1:
输入: [1,4,3,2]
输出: 4
解释: n 等于 2, 最大总和为 4 = min(1, 2) + min(3, 4).
*/
int arrayPairSum(vector<int>& nums) {
	sort(nums.begin(),nums.end());

	int cnt=0;
	for(size_t i=0;i<nums.size();i+=2){
		cnt+=nums[i];
	}

	return cnt;
}

/*
563. 二叉树的坡度

给定一个二叉树，计算整个树的坡度。
一个树的节点的坡度定义即为，该节点左子树的结点之和和右子树结点之和的差的绝对值。空结点的的坡度是0。
整个树的坡度就是其所有节点的坡度之和。

示例:
输入: 
         1
       /   \
      2     3
输出: 1
解释: 
结点的坡度 2 : 0
结点的坡度 3 : 0
结点的坡度 1 : |2-3| = 1
树的坡度 : 0 + 0 + 1 = 1
*/
int FindTiltCore(TreeNode* root, int& res) {
    if (root == NULL) {
        return 0;
    }
    int left = FindTiltCore(root->left, res);
    int right = FindTiltCore(root->right, res);
    res += abs(left - right);
    return left + right + root->val;
}

int findTilt(TreeNode* root) {
    if (root == NULL) {
        return 0;
    }
    int res = 0;
    FindTiltCore(root, res);
    return res;
}



/*
566. 重塑矩阵

在MATLAB中，有一个非常有用的函数 reshape，它可以将一个矩阵重塑为另一个大小不同的新矩阵，但保留其原始数据。
给出一个由二维数组表示的矩阵，以及两个正整数r和c，分别表示想要的重构的矩阵的行数和列数。
重构后的矩阵需要将原始矩阵的所有元素以相同的行遍历顺序填充。
如果具有给定参数的reshape操作是可行且合理的，则输出新的重塑矩阵；否则，输出原始矩阵。

示例 1:
输入: 
nums = 
[[1,2],
 [3,4]]
r = 1, c = 4
输出: 
[[1,2,3,4]]
解释:
行遍历nums的结果是 [1,2,3,4]。新的矩阵是 1 * 4 矩阵, 用之前的元素值一行一行填充新矩阵。
*/
vector<vector<int>> matrixReshape(vector<vector<int>>& nums, int r, int c) {
	int rows=nums.size();
	int cols=nums[0].size();

	if(rows*cols != r*c){
		return nums;
	}

	std::vector<std::vector<int>> vres;
	for(int i=0;i<r;++i){
		std::vector<int> v;
		for(int j=0;j<c;++j){
			int n=i*c+j;
			v.push_back(nums[n/cols][n%cols]);
		}

		vres.push_back(v);
	}

	return vres;
}

/*
572. 另一个树的子树

给定两个非空二叉树 s 和 t，检验 s 中是否包含和 t 具有相同结构和节点值的子树。
s 的一个子树包括 s 的一个节点和这个节点的所有子孙。s 也可以看做它自身的一棵子树。

示例 1:
给定的树 s:

     3
    / \
   4   5
  / \
 1   2
给定的树 t：
   4 
  / \
 1   2
返回 true，因为 t 与 s 的一个子树拥有相同的结构和节点值。。
*/
bool DoesTreeHavaTree(TreeNode *s, TreeNode *t) {
    if (!s && !t) {
        return true;
    }
    if (!s || !t) {
        return false;
    }
    if (s->val != t->val) {
        return false;
    }
    return DoesTreeHavaTree(s->left, t->left) && DoesTreeHavaTree(s->right, t->right);
}

bool isSubtree(TreeNode* s, TreeNode* t) {        
    bool flag = false;
    if (s != nullptr && t != nullptr) {
        if (s->val == t->val) {
            flag = DoesTreeHavaTree(s, t);
        }
        if (!flag) {
           flag = isSubtree(s->left, t);
        } 
        if (!flag) {
            flag = isSubtree(s->right, t);
        }
    }        
    return flag;
}

/*
575. 分糖果

给定一个偶数长度的数组，其中不同的数字代表着不同种类的糖果，每一个数字代表一个糖果。
你需要把这些糖果平均分给一个弟弟和一个妹妹。返回妹妹可以获得的最大糖果的种类数。

示例 1:
输入: candies = [1,1,2,2,3,3]
输出: 3
解析: 一共有三种种类的糖果，每一种都有两个。
     最优分配方案：妹妹获得[1,2,3],弟弟也获得[1,2,3]。这样使妹妹获得糖果的种类数最多。
*/
int distributeCandies(vector<int>& candies) {
	     int count = 0;
        int tmp[200001] = {0};
        for(size_t i = 0; i < candies.size(); ++i)
        {
            if(tmp[candies[i] + 100000] == 0) 
            {
                tmp[candies[i] + 100000] = 1;
                ++count;
            }
        }
        return min(count, (int)candies.size()/2);
}

/*
581. 最短无序连续子数组

给定一个整数数组，你需要寻找一个连续的子数组，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。
你找到的子数组应是最短的，请输出它的长度。

示例 1:
输入: [2, 6, 4, 8, 10, 9, 15]
输出: 5
解释: 你只需要对 [6, 4, 8, 10, 9] 进行升序排序，那么整个表都会变为升序排序。
*/
int findUnsortedSubarray(vector<int>& nums) {
        int left=0, right=0;
        int lenth = nums.size();
        int lMax=INT_MIN; // lMax是前i-1个元素的最大值
        int rMin=INT_MAX;
        for(int i=0;i<lenth;i++){
            if(nums[i]<lMax){  // 如果该元素小于前面所有元素的最大值, 则该元素属于需要重排的区间
                right = i;
            }
            else{ 
                lMax = nums[i];
            }
        }        

        for(int i=lenth-1;i>=0;i--){
            if(nums[i]>rMin){
                left = i;
            }
            else{
                rMin = nums[i];
            }
        }
        return right==left? 0: right-left+1;
}

/*
589. N叉树的前序遍历

给定一个 N 叉树，返回其节点值的前序遍历。
例如，给定一个 3叉树 :
返回其前序遍历: [1,3,5,6,2,4]。
*/
// class Node {
// public:
//     int val;
//     vector<Node*> children;

//     Node() {}

//     Node(int _val) {
//         val = _val;
//     }

//     Node(int _val, vector<Node*> _children) {
//         val = _val;
//         children = _children;
//     }
// };

vector<int> preorder(Node* root) {
	 vector< int > ans ;
        stack <Node *> slist ;
        if( root == NULL )
            return ans ;
        slist.push( root ) ;
        while( slist.size() ){
            Node *top = slist.top() ;
            slist.pop();
            ans.push_back( top->val ) ;
            for( int i=top->children.size()-1;i>=0 ; i--){
                slist.push(top->children[i]);
            }
        }
        return ans ;   
}

/*
590. N叉树的后序遍历

给定一个 N 叉树，返回其节点值的后序遍历。
例如，给定一个 3叉树 :
 
返回其后序遍历: [5,6,3,2,4,1].
*/
vector<int> postorder(Node* root) {
	vector< int > ans ;
        stack <Node *> slist ;
        if( root == NULL )
            return ans ;
        slist.push( root ) ;
        while( slist.size() ){
            Node *top = slist.top() ;
            slist.pop();
            ans.push_back( top->val ) ;
            for( size_t i=0;i<top->children.size(); i++){
                slist.push(top->children[i]);
            }
        }

	reverse(ans.begin(),ans.end());

        return ans ; 
}

/*
594. 最长和谐子序列

和谐数组是指一个数组里元素的最大值和最小值之间的差别正好是1。
现在，给定一个整数数组，你需要在所有可能的子序列中找到最长的和谐子序列的长度。

示例 1:
输入: [1,3,2,2,5,2,3,7]
输出: 5
原因: 最长的和谐数组是：[3,2,2,2,3].
*/
int findLHS(vector<int>& nums) {
	int res=0;
	map<int,int> conters;
	for(size_t i=0;i<nums.size();++i){
		conters[nums[i]]++;
	}

	for(auto p:conters){
		int tmp=0;
		if(conters.count(p.first+1)>0){
			tmp=conters[p.first+1];
		}
		if(conters.count(p.first-1)>0){
			tmp=conters[p.first-1];
		}
		if(tmp>0){
			res=max(res,tmp+p.second);
		}
		
	}

	return res;
}

/*
598. 范围求和 II

给定一个初始元素全部为 0，大小为 m*n 的矩阵 M 以及在 M 上的一系列更新操作。
操作用二维数组表示，其中的每个操作用一个含有两个正整数 a 和 b 的数组表示，
含义是将所有符合 0 <= i < a 以及 0 <= j < b 的元素 M[i][j] 的值都增加 1。
在执行给定的一系列操作后，你需要返回矩阵中含有最大整数的元素个数。

示例 1:
输入: 
m = 3, n = 3
operations = [[2,2],[3,3]]
输出: 4
解释: 
初始状态, M = 
[[0, 0, 0],
 [0, 0, 0],
 [0, 0, 0]]

执行完操作 [2,2] 后, M = 
[[1, 1, 0],
 [1, 1, 0],
 [0, 0, 0]]

执行完操作 [3,3] 后, M = 
[[2, 2, 1],
 [2, 2, 1],
 [1, 1, 1]]

M 中最大的整数是 2, 而且 M 中有4个值为2的元素。因此返回 4。
*/
int maxCount(int m, int n, vector<vector<int>>& ops) {
	if(ops.empty()){
		return m*n;
	}

	int min_x=INT_MAX;
	int min_y=INT_MAX;

	for(size_t i=0;i<ops.size();++i){
		min_x=min_x>ops[i][0]?ops[i][0]:min_x;
		min_y=min_y>ops[i][1]?ops[i][1]:min_y;
	}

	return min_x*min_y;
}

/*
599. 两个列表的最小索引总和

假设Andy和Doris想在晚餐时选择一家餐厅，并且他们都有一个表示最喜爱餐厅的列表，每个餐厅的名字用字符串表示。
你需要帮助他们用最少的索引和找出他们共同喜爱的餐厅。 如果答案不止一个，则输出所有答案并且不考虑顺序。 你可以假设总是存在一个答案。

示例 1:
输入:
["Shogun", "Tapioca Express", "Burger King", "KFC"]
["Piatti", "The Grill at Torrey Pines", "Hungry Hunter Steakhouse", "Shogun"]
输出: ["Shogun"]
解释: 他们唯一共同喜爱的餐厅是“Shogun”。
*/
vector<string> findRestaurant(vector<string>& list1, vector<string>& list2) {
	std::vector<string> v;
	if(list2.empty() || list1.empty()){
		return v;
	}

	map<string,int> counters;
	for(size_t i=0;i<list1.size();++i){
		counters[list1[i]]=i;
	}

	int res=INT_MAX;
	for(size_t i=0;i<list2.size();++i){
		if(counters.count(list2[i])>0){
			int temp=i+counters[list2[i]];
			if(res>temp){
				res=temp;
				if(!v.empty()){
					v.clear();
				}
				v.push_back(list2[i]);
			}else if(res==temp){
				v.push_back(list2[i]);
			}
		}
	}

	return v;
}

/*
605. 种花问题

假设你有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花卉不能种植在相邻的地块上，它们会争夺水源，两者都会死去。
给定一个花坛（表示为一个数组包含0和1，其中0表示没种植花，1表示种植了花），和一个数 n 。
能否在不打破种植规则的情况下种入 n 朵花？能则返回True，不能则返回False。

示例 1:
输入: flowerbed = [1,0,0,0,1], n = 1
输出: True
*/
bool canPlaceFlowers(vector<int>& flowerbed, int n) {
	if(flowerbed.empty()){
		return false;
	}

	flowerbed.push_back(0);
	flowerbed.insert(flowerbed.begin(),0);
	int num=0;
	for(size_t i=0;i<flowerbed.size()-2;++i){
		if(flowerbed[i]==0 && flowerbed[i+1]==0 && flowerbed[i+2]==0){
			num++;
			i++;
		}
	}

	return num>=n;
}

/*
606. 根据二叉树创建字符串

你需要采用前序遍历的方式，将一个二叉树转换成一个由括号和整数组成的字符串。
空节点则用一对空括号 "()" 表示。而且你需要省略所有不影响字符串与原始二叉树之间的一对一映射关系的空括号对。

示例 1:
输入: 二叉树: [1,2,3,4]
       1
     /   \
    2     3
   /    
  4     
输出: "1(2(4))(3)"
解释: 原本将是“1(2(4)())(3())”，
在你省略所有不必要的空括号对之后，
它将是“1(2(4))(3)”。
*/
string tree2str(TreeNode* t) {
	if(t==NULL){
		return "";
	}

	string s=to_string(t->val);
	if(t->left!=NULL && t->right!=NULL){
		s+="("+tree2str(t->left)+")";
		s+="("+tree2str(t->right)+")";
	}
	if(t->left==NULL && t->right !=NULL){
		s+="()";
		s+="("+tree2str(t->right)+")";
	}
	if(t->left!=NULL && t->right==NULL){
		s+="("+tree2str(t->left)+")";
	}

	return s;
}

/*
617. 合并二叉树

给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，
否则不为 NULL 的节点将直接作为新二叉树的节点。

示例 1:
输入: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
输出: 
合并后的树:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7

注意: 合并必须从两个树的根节点开始。
*/
TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
	if(t1==NULL){
		return t2;
	}
	if(t2==NULL){
		return t1;
	}

	if(t1!=NULL && t2!=NULL){
		t1->val+=t2->val;
	} 
	
	t1->left=mergeTrees(t1->left,t2->left);
	t1->right=mergeTrees(t1->right,t2->right);

	return t1;
}

/*
628. 三个数的最大乘积

给定一个整型数组，在数组中找出由三个数组成的最大乘积，并输出这个乘积。

示例 1:
输入: [1,2,3] 输出: 6
*/
int maximumProduct(vector<int>& nums) {
	sort(nums.begin(),nums.end(),[](int a,int b){return a>b;});

	return max(nums[0]*nums[1]*nums[2],nums[nums.size()-1]*nums[nums.size()-2]*nums[0]);
}

/*
633. 平方数之和

给定一个非负整数 c ，你要判断是否存在两个整数 a 和 b，使得 a2 + b2 = c。

示例1:
输入: 5 输出: True
解释: 1 * 1 + 2 * 2 = 5
*/
bool judgeSquareSum(int c) {
	int i=0;
	double x=sqrt(c);
	int j=(int)x;
	while(i<=j){
		if((c-i*i)==j*j){
			return true;
		}else if((c-i*i)<j*j){
			--j;
		}else{
			++i;
		}
	}

	return false;
}

/*
637. 二叉树的层平均值

给定一个非空二叉树, 返回一个由每层节点平均值组成的数组.

示例 1:
输入:
    3
   / \
  9  20
    /  \
   15   7
输出: [3, 14.5, 11]
解释:第0层的平均值是 3,  第1层是 14.5, 第2层是 11. 因此返回 [3, 14.5, 11].
*/
 /* struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };*/
vector<double> averageOfLevels(TreeNode* root) {
	queue<TreeNode *> q1;
	queue<TreeNode *> q2;
	std::vector<double> v;

	q1.push(root);
	long long tmp=0;
	int n=0;
	while(!q1.empty()){
		TreeNode* node=q1.front();
		tmp+=node->val;
		++n;
		q1.pop();

		if(node->left != NULL){
			q2.push(node->left);
		}
		if(node->right!=NULL){
			q2.push(node->right);
		}

		if(q1.empty()){
			v.push_back((double)tmp/n);

			q1=q2;
			tmp=0;
			n=0;
			q2=queue<TreeNode* >();
		}
	}

	return v;
}

/*
643. 子数组最大平均数 I

给定 n 个整数，找出平均数最大且长度为 k 的连续子数组，并输出该最大平均数。

示例 1:
输入: [1,12,-5,-6,50,3], k = 4
输出: 12.75
解释: 最大平均数 (12-5-6+50)/4 = 51/4 = 12.75
*/
double findMaxAverage(vector<int>& nums, int k) {
	if(nums.empty() || k>(int)nums.size()||k<=0){
		return 0;
	}
	double max=INT_MIN;
	int sum=0;
	for(int i=0;i< (int)nums.size();++i){
		sum+=nums[i];

		if(i>=k-1){
			if(i>k-1){
				sum-=nums[i-k];
			}

			double avg=(double)sum/k;
			if(avg>max){
				max=avg;
			}
		}
	}

	return max;
}

/*
645. 错误的集合

集合 S 包含从1到 n 的整数。不幸的是，因为数据错误，导致集合里面某一个元素复制了成了
集合里面的另外一个元素的值，导致集合丢失了一个整数并且有一个元素重复。
给定一个数组 nums 代表了集合 S 发生错误后的结果。你的任务是首先寻找到重复出现的整数，
再找到丢失的整数，将它们以数组的形式返回。

示例 1:
输入: nums = [1,2,2,4]
输出: [2,3]
*/
vector<int> findErrorNums(vector<int>& nums) {
	std::vector<int> v;
	set<int> s(nums.begin(),nums.end());

	int sum=(nums.size()*(nums.size()+1))/2;
	int n=accumulate(nums.begin(),nums.end(),0);
	int l=accumulate(s.begin(),s.end(),0);
	v.push_back(n-l);
	v.push_back(sum-l);

	return v;
}

/*
653. 两数之和 IV - 输入 BST

给定一个二叉搜索树和一个目标结果，如果 BST 中存在两个元素且它们的和等于给定的目标结果，则返回 true。

案例 1:
输入: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 9
输出: True
*/
void findTarget_inorder(TreeNode *root,std::vector<int> &v){
	if(root==NULL){
		return;
	}

	findTarget_inorder(root->left,v);
	v.push_back(root->val);
	findTarget_inorder(root->right,v);
}

bool findTarget(TreeNode* root, int k) {
	std::vector<int> v;

	findTarget_inorder(root,v);

	size_t i=0,j=v.size()-1;
	while(i<j){
		if(v[i]+v[j]<k){
			++i;
		}else if(v[i]+v[j]>k){
			--j;
		}else{
			return true;
		}
	}

	return false;
}

/*
657. 机器人能否返回原点

在二维平面上，有一个机器人从原点 (0, 0) 开始。给出它的移动顺序，
判断这个机器人在完成移动后是否在 (0, 0) 处结束。
移动顺序由字符串表示。字符 move[i] 表示其第 i 次移动。机器人的有效动作有 R（右），L（左），
U（上）和 D（下）。如果机器人在完成所有动作后返回原点，则返回 true。否则，返回 false。
注意：机器人“面朝”的方向无关紧要。 “R” 将始终使机器人向右移动一次，
“L” 将始终向左移动等。此外，假设每次移动机器人的移动幅度相同。

示例 1:
输入: "UD" 输出: true
解释：机器人向上移动一次，然后向下移动一次。所有动作都具有相同的幅度，
因此它最终回到它开始的原点。因此，我们返回 true。
*/
bool judgeCircle(string moves) {
	int left_right=0;
	int up_down=0;
	for(size_t i=0;i<moves.size();++i){
		if(moves[i]=='U'){
			++up_down;
		}else if(moves[i]=='D'){
			--up_down;
		}else if(moves[i]=='L'){
			++left_right;
		}else{
			--left_right;
		}
	}

	return left_right==0 && up_down==0;
}

/*
661. 图片平滑器

包含整数的二维矩阵 M 表示一个图片的灰度。你需要设计一个平滑器来让每一个单元的灰度成为平均灰度 (向下舍入) ，
平均灰度的计算是周围的8个单元和它本身的值求平均，如果周围的单元格不足八个，则尽可能多的利用它们。

示例 1:
输入:
[[1,1,1],
 [1,0,1],
 [1,1,1]]
输出:
[[0, 0, 0],
 [0, 0, 0],
 [0, 0, 0]]
解释:
对于点 (0,0), (0,2), (2,0), (2,2): 平均(3/4) = 平均(0.75) = 0
对于点 (0,1), (1,0), (1,2), (2,1): 平均(5/6) = 平均(0.83333333) = 0
对于点 (1,1): 平均(8/9) = 平均(0.88888889) = 0
*/
vector<vector<int>> imageSmoother(vector<vector<int>>& M) {
	if( M.size() == 0){
		return std::vector<vector<int>>();
	}

	size_t rows=M.size();
	size_t cols=M[0].size();
	vector<vector<int>> res( rows, vector<int>(cols, 0));

	for(size_t i=0;i<rows;++i){
		for(size_t j =0;j<cols;++j){
			int sum=0;
			int count=0;
			for(int m=-1;m<2;++m){
				for(int n=-1;n<2;++n){
					if(i+m>=0 && j+n>=0 && i+m<rows && j+n<cols){
						sum+=M[m+i][n+j];
						++count;
					}
				}
			}

			res[i][j]=sum/count;
		}
	}

	return res;
}

/*
665. 非递减数列

给定一个长度为 n 的整数数组，你的任务是判断在最多改变 1 个元素的情况下，该数组能否变成一个非递减数列。
我们是这样定义一个非递减数列的： 对于数组中所有的 i (1 <= i < n)，满足 array[i] <= array[i + 1]。

示例 1:
输入: [4,2,3]
输出: True
解释: 你可以通过把第一个4变成1来使得它成为一个非递减数列。
*/
bool checkPossibility(vector<int>& nums) {
	int flag = 1, nCount = nums.size();
        for (int i = 1; i < nCount; ++i) 
        {
            if (nums[i] < nums[i - 1]) 
            {
                if (flag == 0) 
                    return false;
                if (i == 1 || nums[i] >= nums[i - 2]) 
                    nums[i - 1] = nums[i];
                else 
                    nums[i] = nums[i - 1];
                flag--;
            } 
        }
        return true;
}

/*
669. 修剪二叉搜索树

给定一个二叉搜索树，同时给定最小边界L 和最大边界 R。通过修剪二叉搜索树，
使得所有节点的值在[L, R]中 (R>=L) 。你可能需要改变树的根节点，所以结果应当返回修剪好的二叉搜索树的新的根节点。

示例 1:
输入: 
    1
   / \
  0   2

  L = 1
  R = 2
输出: 
    1
      \
       2
*/
TreeNode* trimBST(TreeNode* root, int L, int R) {
	    if(!root ) return nullptr;
        //剪枝分三种情况 0 [1,2, 3,] 2 [1,2,3] 4 [1,2,3]
        // 0 [1,2,3] ,左枝都不要了 root不要了，root=右枝
        if ( root->val < L) {
            root->left = nullptr;
            root = trimBST(root->right,L,R);
            return root;
        }
        // 2 [1,2,3];
        if ( root->val >= L && root->val <= R) {
            root->left = trimBST(root->left,L,R);
            root->right = trimBST(root->right,L,R);
            return root;
        }
        // 4 [1,2,3];
        if ( root->val > R) {
            root->right  = nullptr;
            root = trimBST(root->left,L,R);
            return root;
        }
        
        return root;  
}

/*
671. 二叉树中第二小的节点

给定一个非空特殊的二叉树，每个节点都是正数，并且每个节点的子节点数量只能为 2 或 0。
如果一个节点有两个子节点的话，那么这个节点的值不大于它的子节点的值。 
给出这样的一个二叉树，你需要输出所有节点中的第二小的值。如果第二小的值不存在的话，输出 -1 。

示例 1:
输入: 
    2
   / \
  2   5
     / \
    5   7

输出: 5
说明: 最小的值是 2 ，第二小的值是 5 。
*/
int findSecondMinimumValue(TreeNode* root) {
	if(root==NULL || (root->left ==NULL && root->right==NULL)){
		return -1;
	}

	int left=root->left->val;
	int right=root->right->val;

	if(left==root->val){
		left=findSecondMinimumValue(root->left);
	}

	if(right==root->val){
		right=findSecondMinimumValue(root->right);
	}


	if(left!=-1 && right!=-1){
		return min(left,right);
	}

	if(left!=-1){
		return left;
	}

	return right;
}

/*
674. 最长连续递增序列

给定一个未经排序的整数数组，找到最长且连续的的递增序列。

示例 1:
输入: [1,3,5,4,7]
输出: 3
解释: 最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为5和7在原数组里被4隔开。 
*/
int findLengthOfLCIS(vector<int>& nums) {
	if(nums.empty()){
		return 0;
	}

	size_t preIndex=0;
	size_t curIndex=0;
	size_t maxLength=0;

	while(curIndex<nums.size()-1){
		if(nums[curIndex]>=nums[curIndex+1]){
			if(maxLength<(curIndex-preIndex+1)){
				maxLength=curIndex-preIndex+1;
			}
			++curIndex;
			preIndex=curIndex;
		}else{
			++curIndex;
		}
	}

	if(maxLength<curIndex-preIndex){
		maxLength=curIndex-preIndex+1;
	}

	return maxLength;
}

/*
680. 验证回文字符串 Ⅱ

给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。
示例 1:
输入: "aba" 输出: True
*/
bool validPalindrome(string s) {
 if(s.length()==1)
            return 1;
        int i=0,j=s.length()-1,flag=0;
        while(i<j)
        {
            if(s[i] != s[j])
            {
                if(flag==0)
                    flag=1;
                else
                    return 0;

                if(i+1==j)
                    return 1;
                else
                {
                    if(s[i+1]==s[j] &&s[i+2]==s[j-1])
                        i++;
                    else
                        j--;
                }
            }
            else
            {
                i++;
                j--;
            }
        }

        return 1;
}

/*
682. 棒球比赛

你现在是棒球比赛记录员。
给定一个字符串列表，每个字符串可以是以下四种类型之一：
1.整数（一轮的得分）：直接表示您在本轮中获得的积分数。
2. "+"（一轮的得分）：表示本轮获得的得分是前两轮有效 回合得分的总和。
3. "D"（一轮的得分）：表示本轮获得的得分是前一轮有效 回合得分的两倍。
4. "C"（一个操作，这不是一个回合的分数）：表示您获得的最后一个有效 回合的分数是无效的，应该被移除。
每一轮的操作都是永久性的，可能会对前一轮和后一轮产生影响。
你需要返回你在所有回合中得分的总和。

示例 1:
输入: ["5","2","C","D","+"]
输出: 30
解释: 
第1轮：你可以得到5分。总和是：5。
第2轮：你可以得到2分。总和是：7。
操作1：第2轮的数据无效。总和是：5。
第3轮：你可以得到10分（第2轮的数据已被删除）。总数是：15。
第4轮：你可以得到5 + 10 = 15分。总数是：30。
*/
int calPoints(vector<string>& ops) {
	vector<int> stack;
        for(size_t i=0;i<ops.size();i++){
            if(ops[i]=="+"){
                stack.push_back(stack[stack.size()-1]+stack[stack.size()-2]);
            }else if(ops[i]=="D"){
                stack.push_back(stack[stack.size()-1]*2);
            }else if(ops[i]=="C"){
                stack.pop_back();
            }else{
                stack.push_back(stoi(ops[i]));
            }
        }
        int sum=0;
        for(auto v:stack){
            sum+=v;
        }
        return sum;
}

/*
686. 重复叠加字符串匹配

给定两个字符串 A 和 B, 寻找重复叠加字符串A的最小次数，
使得字符串B成为叠加后的字符串A的子串，如果不存在则返回 -1。

举个例子，A = "abcd"，B = "cdabcdab"。
答案为 3， 因为 A 重复叠加三遍后为 “abcdabcdabcd”，此时 B 是其子串；
A 重复叠加两遍后为"abcdabcd"，B 并不是其子串。
*/
int repeatedStringMatch(string A, string B) {
	int cnt=0;
	string s=A;
	while(s.length()<B.length()){
		++cnt;
		s+=A;
	}

	if(s.find(B)!=s.npos){
		return cnt;
	}else{
		++cnt;
		s+=A;
		if(s.find(B)!=s.npos){
			return cnt;
		}else{
			return -1;
		}
	}
}

/*
687. 最长同值路径

给定一个二叉树，找到最长的路径，这个路径中的每个节点具有相同值。
这条路径可以经过也可以不经过根节点。
注意：两个节点之间的路径长度由它们之间的边数表示。

示例 1:
输入:
              5
             / \
            4   5
           / \   \
          1   1   5
输出:2
*/
int longestUnivaluePath_dfs(TreeNode *root,int &maxLength){
	if(root==NULL){
		return 0;
	}

	int left=longestUnivaluePath_dfs(root->left,maxLength);
	int right=longestUnivaluePath_dfs(root->right,maxLength);

	if(root->left!=NULL && root->val==root->left->val){
		left+=1;
	}else{
		left=0;
	}

	if(root->right!=NULL && root->val==root->right->val){
		right+=1;
	}else{
		right=0;
	}

	maxLength=max(maxLength,left+right);

	return max(left,right);
}

int longestUnivaluePath(TreeNode* root) {
	if(root==NULL){
		return 0;
	}

	int maxLength=0;
	longestUnivaluePath_dfs(root,maxLength);

	return maxLength;
}

/*
690. 员工的重要性

给定一个保存员工信息的数据结构，它包含了员工唯一的id，重要度 和 直系下属的id。
比如，员工1是员工2的领导，员工2是员工3的领导。他们相应的重要度为15, 10, 5。那么员工1的数据结构是[1, 15, [2]]，
员工2的数据结构是[2, 10, [3]]，员工3的数据结构是[3, 5, []]。注意虽然员工3也是员工1的一个下属，但是由于并不是直系下属，
因此没有体现在员工1的数据结构中。现在输入一个公司的所有员工信息，以及单个员工id，返回这个员工和他所有下属的重要度之和。

示例 1:
输入: [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
输出: 11
解释:员工1自身的重要度是5，他有两个直系下属2和3，而且2和3的重要度均为3。因此员工1的总重要度是 5 + 3 + 3 = 11。
*/
class Employee {
public:
    // It's the unique ID of each node.
    // unique id of this employee
    int id;
    // the importance value of this employee
    int importance;
    // the id of direct subordinates
    vector<int> subordinates;
};

int getImportance(vector<Employee*> employees, int id) {
	if(employees.empty()){
		return 0;
	}

	Employee *p=NULL;
	for(size_t i=0;i<employees.size();++i){
		if(employees[i]->id==id){
			p=employees[i];
		}
	}

	if(p==NULL){
		return 0;
	}

	int sum=p->importance;
	for(size_t i=0;i<p->subordinates.size();++i){
		sum+=getImportance(employees,p->subordinates[i]);
	}

	return sum;
}

int main(){
	// string s="abc";
	// string t="accvbyhgc";
	// cout<<isSubsequence(s,t)<<endl;

	// vector<string> v=readBinaryWatch(1);
	// for(auto c:v){
	// 	cout<<c<<" ";
	// }
	// cout<<endl;

	// cout<<toHex(28)<<endl;

	// string s="aaaabbbb8jgfjkkkkk";
	// cout<<longestPalindrome(s)<<endl;

	// vector<string> res=fizzBuzz(15);
	// for(size_t i=0;i<res.size();++i){
	// 	cout<<res[i]<<endl;
	// }

	// std::vector<int> v={1,2};
	// cout<<thirdMax(v)<<endl;

	// string num1="18582506933032752";
	// string num2="366213329703";
	// cout<<addStrings(num1,num2)<<endl;

	// string s="Hello, my name is John";
	// cout<<countSegments(s)<<endl;

	// cout<<arrangeCoins(1804289383)<<endl;

	// std::vector<int> v={4,3,2,7,8,2,3,1};
	// std::vector<int> res=findDisappearedNumbers(v);
	// for(auto c:res){
	// 	cout<<c<<" ";
	// }
	// cout<<endl;

	// cout<<hammingDistance(1,4)<<endl;;

	// cout<<findComplement(2147483647)<<endl;

	// std::vector<int> v={1};
	// cout<<findMaxConsecutiveOnes(v)<<endl;

	// cout<<convertToBase7(-8)<<endl;

	// cout<<checkPerfectNumber(28)<<endl;

	// cout<<fib(3)<<endl;

	// string s="abcdefg";
	// cout<<reverseStr(s,2)<<endl;

	// string s="LLLALL";
	// cout<<checkRecord(s)<<endl;

	// string s="Let's take LeetCode contest";
	// cout<<reverseWords(s)<<endl;

	// std::vector<int> v={1,1,1,1};
	// cout<<findLHS(v)<<endl;

	// std::vector<int> v={0};
	// cout<<canPlaceFlowers(v,1)<<endl;

	// cout<<judgeSquareSum(2147482647)<<endl;

	string s="aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga";
	cout<<validPalindrome(s)<<endl;

	return 0;
}