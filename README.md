[LeetCode 300. 最长上升子序列 longest increasing subsequence](https://leetcode-cn.com/problems/longest-increasing-subsequence)

## 题目描述
给定一个无序的整数数组，找到其中最长上升子序列的长度。

### 示例:
```
输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
```

### 说明:
* 可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。  
* 你算法的时间复杂度应该为 O(n2)。    

**进阶:** 你能将算法的时间复杂度降低到 O(n log n) 吗? 

## 解题思路
看完题目描述，我还以为是要找的这个子序列是连续元素组成，直到看到示例，才发现这个子序列可以不连续。如果要求的子序列是连续的话，这题就简单多了，直接遍历一遍，比较相邻元素的大小即可。  
那么对于可以不连续的子序列来说，问题好像没那么简单了。  
首先能想到的最直接的方法，就是暴力穷举的方式，但是复杂度比较高，这里就不具体说了。  
下面看下动态规划解法。  

## 动态规划
我们再来分析下示例 [10,9,2,5,3,7,101,18] ，其实针对每个元素来说，它及之前元素组成的数组中的最长上升子序列都是确定了的，也就是说具有最优子结构，这样我们就可以从后往前逐步计算出整个数组的最长上升子序列。  
分两步走：  
1 状态的定义：定义DP[i]，表示第i个元素及其前面元素组成的子数组(下标0-i)中，最长的上升子序列的长度。
2 状态转移方程：在第i个元素之前的所有子数组中，对于j(j<i)，第j个元素及之前的元素组成的子数组中上升子序列长度就是DP[j]，那么对于nums[j]<nums[i]时,则DP[i]=max{DP[j]+1}，否则DP[i]不变。

这里做了两重循环，所以复杂度为O(N²)，空间复杂度也是O(N²)。

c++代码实现如下：  
```C++
int lengthOfLIS(vector<int>& nums) {
    int vec_len = nums.size();
    if(vec_len < 2) //数组长度为0，返回0；数组长度为1，则最长上升子序列就是1
        return vec_len;

    int max_len = 1;
    vector<int> DP(vec_len, 1); //存放第i个元素及之前的，所有上升子序列的长度，初始都是1

    for(int i = 1; i < vec_len; i++){ //逐个元素，计算其前面所有元素组成的数组中，最长的上升子序列长度
        for(int j = 0; j < i; j++){
            if(nums[j] < nums[i]){ //因为是按递增来算长度
                DP[i] = std::max(DP[i], DP[j] + 1); //第i个元素及其前面元素中，最长子序列长度就是
                                                          //取其前面元素中最大的+1与当前最大的长度中最大的
            }                 
            //else if(nums[j] >= nums[i]){} 这里就不是上升序列了，所以长度不变
        }
        max_len = std::max(max_len, DP[i]);
    }

    return max_len;
}
```   


## 改进版
这个是我看到题解中的解法，很巧妙的解法，它把上面动态规划的解法中，第二重循环做了优化。它是维护了一个数组，用来暂存了可能成为最长上升子序列的所有元素，最后这个数组的长度就是要求的最长上升子序列的长度，具体做法如下：
逐个的取出原数组nums中元素，往临时数组tmp_lis中放，若该元素a大于tmp_lis中所有元素，则追加到tmp_lis最后；若不是，则找出tmp_lis中大于该元素a的最小的元素b，然后用a替换tmp_lis中的这个元素b，直到nums数组遍历完毕。

我们来看下，对于示例数组：
[10,9,2,5,3,7,101,18]
它是逐个的遍历数组元素，往另一个数组中放，假设这个数组为tmp_lis[]。  

取第一个元素10：  
tmp_lis[] = {10}  
取第二个元素9，因为9小于10，则用9替换10：  
tmp_lis[] = {9}  
取第三个元素2，因为2  
tmp_lis[] = {2}  
取第四个元素5，因为5>2,所以追加到tmp_lis中   
tmp_lis[] = {2,5}  
取第五个元素3，因为2<3<5,所以用3替换5  
tmp_lis[] = {2,3}  
取第六个元素7，因为7>3>2,所以直接追加到tmp_lis中  
tmp_lis[] = {2,3,7}  
取第七个元素101，因为101>7>3>2,所以直接追加到tmp_lis中  
tmp_lis[] = {2,3,7,101}  
取第八个元素18，因为18<101,18>7,所以用18替换101  
tmp_lis[] = {2,3,7,18}  
所以最终最长上升子序列的长度就是tmp_lis的长度，为4。  

对于查找tmp_lis中比nums[i]大的最小元素，我们采用二分法查找，所以这层循环的复杂度是O(logN)，加上外层的O(N)的复杂度，最终算法复杂度就是O(NlogN)。  
**注意:** tmp_lis最后并非就是最长上升子序列，它只是暂存了曾经是最长上升子序列的数组，可以自行以[6,7,8,9,1,2,3]为例自行验算下。

c++代码实现如下：  
```C++
int lengthOfLIS(vector<int>& nums) {
    vector<int> tmp_lis;
    for(int i = 0; i < nums.size(); i++){
        auto it = std::lower_bound(tmp_lis.begin(), tmp_lis.end(), nums[i]);/二分查找
        if(it != tmp_lis.end())
            *it = nums[i];
        else
            tmp_lis.push_back(nums[i]);//因为tmp_lis已经有序，所以找不到比*it大的数据的时候，则*it比tmp_lis中所有元素都大，直接追加
    }

    return tmp_lis.size();
}
```   

