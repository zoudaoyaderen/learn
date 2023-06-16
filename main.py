import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_dim, config.n_dim * 3)
        self.c_proj = nn.Linear(config.n_dim, config.n_dim)
        self.attn_drop = nn.Dropout(config.att_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        m_len = config.max_len
        self.register_buffer("bias", torch.tril(torch.ones(m_len, m_len)).view(1, 1, m_len, m_len))
        self.n_dim = config.n_dim
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.shape
        nh = self.n_head
        hs = C // nh

        q, k, v = self.c_attn(x).split(C, dim=-1)  # (B, T, C)
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        att = q @ k.transpose(2, 3) * 1.0 / math.sqrt(hs)
        att = att.masked_fill(self.bias == 0, float("-inf"))
        att = F.softmax(att)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_drop(y)

        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_dim)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_dim, config.n_dim * 4),
            act=nn.GELU,
            c_proj=nn.Linear(config.n_dim * 4, config.n_dim),
            drop=nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.drop(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlpf(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_len = config.max_len
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_dim),
                wpe=nn.Embedding(self.max_len, config.n_dim),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block(config) for _ in config.n_layer]),
                ln=nn.LayerNorm(config.n_dim),
            )
        )
        self.lm_head = nn.Linear(config.n_dim, config.vocab_size)

    def forward(self, idx, targets):
        device = idx.device
        B, T = idx.shape
        pos = torch.arange(0, self.max_len, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        wte = self.transformer.wte(idx)
        wpe = self.transformer.wpe(pos)
        x = self.transformer.drop(wte + wpe)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln(x)

        logits = self.lm_head(x)  # (B, T, V)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def QuickSort(nums):
    if len(nums) <= 1:
        return nums
    llist, mlist, rlist, key = [], [], [], nums[0]  # 取 index 0 为 key
    mlist.append(key)
    for i in range(1, len(nums)):
        num = nums[i]
        if num < key:
            llist.append(num)
        elif num > key:
            rlist.append(num)
        else:
            mlist.append(num)
    return QuickSort(llist) + mlist + QuickSort(rlist)


nums = np.random.randint(0, 100, (10,)).tolist()


# print(nums)


# print(QuickSort(nums))


def QuickSortInplace(left, right):
    if left >= right:
        return nums
    l, r, key = left, right, nums[left]
    while (l < r):
        while (l < r) and nums[r] >= key:
            r -= 1
        nums[l] = nums[r]
        while (l < r) and nums[l] < key:
            l += 1
        nums[r] = nums[l]
    nums[l] = key
    QuickSortInplace(left, l - 1)
    QuickSortInplace(l + 1, right)


# QuickSortInplace(0, len(nums) - 1)
# print(nums)


def BubbleSort(nums):
    if len(nums) <= 1:
        return nums
    for i in range(len(nums) - 1, 0, -1):
        flag = False
        for j in range(i):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                flag = True
        if not flag:
            break
    return nums


# print(BubbleSort(nums))

def BinarySearch(nums, num):
    l, r = 0, len(nums) - 1
    while (l <= r):
        mid = (l + r + 1) // 2
        # print(l, mid, r)
        if num > nums[mid]:
            l = mid + 1
        elif num < nums[mid]:
            r = mid - 1
        else:
            return mid
    return -1


# nums = list(range(0, 10, 2))
#
# print(nums)
# print(BinarySearch(nums, 4))


def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + (0 if word1[i - 1] == word2[j - 1] else 1),
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1
            )
    return dp[m][n]


# print(edit_distance("word1", "wo1rd122"))


n = 6
bag_weight = 10
w = [2, 2, 3, 1, 5, 2]
v = [2, 3, 1, 5, 4, 3]


# value = fun(n, bag_weight, w, v)
# show(n, bag_weight, w, value)


def bag(n, bag_weight, w, v):
    value = np.zeros((n + 1, bag_weight + 1))
    for i in range(1, n + 1):
        for j in range(1, bag_weight + 1):
            value[i][j] = value[i - 1][j]  # default
            if j >= w[i - 1]:
                value[i][j] = max(value[i - 1][j], value[i - 1][j - w[i - 1]] + v[i - 1])
                print(value)

    things = []
    j = bag_weight
    for i in range(n, 0, -1):
        if value[i][j] > value[i - 1][j]:
            things.append(i - 1)
            j -= w[i - 1]
    return value[n][bag_weight], things


# print(bag(n, bag_weight, w, v))


def partition(seq):
    pi, seq = seq[0], seq[1:]  # 选取并移除主元
    lo = [x for x in seq if x <= pi]  # 选出小于第一个数的所有元素
    hi = [x for x in seq if x > pi]  ##选出大于第一个数的所有元素
    return lo, pi, hi


def select(seq, k):
    lo, pi, hi = partition(seq)
    m = len(lo)  # 小于第一个数的元素有几个
    if m == k: return pi
    if m < k: return select(hi, k - m - 1)
    return select(lo, k)


seq = (1, 2, 3, 4, 5)


# print(partition(seq))
# print(select(seq, 3))


def MergeSort(nums):
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    llist, rlist = MergeSort(nums[:mid]), MergeSort(nums[mid:])
    res = []
    i, j = 0, 0
    while i < len(llist) and j < len(rlist):
        if llist[i] < rlist[j]:
            res.append(llist[i])
            i += 1
        else:
            res.append(rlist[j])
            j += 1
    res += llist[i:] + rlist[j:]
    return res


nums = np.random.randint(0, 20, (10,)).tolist()
# print(nums)
# print(MergeSort(nums))

nums = np.random.randint(0, 100, (10,)).tolist()


# print(nums)


def Heapify(start, end):
    father = start
    son = father * 2 + 1
    while (son <= end):
        if (son + 1) <= end and nums[son + 1] < nums[son]:
            son += 1
        if nums[father] > nums[son]:
            nums[father], nums[son] = nums[son], nums[father]
            father = son
            son = father * 2 + 1
        else:
            return


def HeapInit():
    for i in range((len(nums) - 1) // 2, -1, -1):
        Heapify(i, len(nums) - 1)


def HeapSort():
    topk = 5
    for i in range(len(nums) - 1, 0, -1):
        nums[i], nums[0] = nums[0], nums[i]
        if i + topk == len(nums):
            break
        Heapify(0, i - 1)


# HeapInit()
# HeapSort()
# print(nums)

nums = np.random.randint(0, 20, (9,)).tolist()


# print(nums)


def quickSelect(nums, k):
    '''

    :param nums:
    :param k:
    :return: klist, rlist
    '''
    llist, mlist, rlist = [], [], []

    if k <= 0 or len(nums) <= 0:
        return llist, mlist, rlist

    pivot = nums[random.randint(0, len(nums) - 1)]
    for i, n in enumerate(nums):
        if n < pivot:
            llist.append(n)
        elif n == pivot:
            mlist.append(n)
        else:
            rlist.append(n)

    if k <= len(rlist):
        _llist, _mlist, _rlist = quickSelect(rlist, k)
        return llist + mlist + _llist, _mlist, _rlist
    elif k <= len(rlist + mlist):
        return llist, mlist, rlist
    else:
        _llist, _mlist, _rlist = quickSelect(llist, k - len(rlist + mlist))
        return _llist, _mlist, _rlist + mlist + rlist


# print(sorted(nums))
# print(quickSelect(nums, 4))


def wiggleSort(nums):
    # nums.sort()
    mid_i = (len(nums) + 1) // 2
    llist, mlist, rlist = quickSelect(nums, mid_i)
    print(llist, mlist, rlist)
    nums[::2], nums[1::2] = (llist + mlist)[:mid_i], (llist + mlist + rlist)[mid_i:]
    return nums


# print(wiggleSort(nums))


def minWindow(self, s: str, t: str) -> str:
    need = collections.defaultdict(int)
    for c in t:
        need[c] += 1
    needCnt = len(t)
    i = 0
    res = (0, float('inf'))
    for j, c in enumerate(s):
        if need[c] > 0:
            needCnt -= 1
        need[c] -= 1
        if needCnt == 0:  # 步骤一：滑动窗口包含了所有T元素
            while True:  # 步骤二：增加i，排除多余元素
                c = s[i]
                if need[c] == 0:
                    break
                need[c] += 1
                i += 1
            if j - i < res[1] - res[0]:  # 记录结果
                res = (i, j)
            need[s[i]] += 1  # 步骤三：i增加一个位置，寻找新的满足条件滑动窗口
            needCnt += 1
            i += 1
    return '' if res[1] > len(s) else s[res[0]:res[1] + 1]  # 如果res始终没被更新过，代表无满足条件的结果


def mySqrt(x):
    # 牛顿迭代 x1 = 0.5 * (x0 + c / x0)
    c, x0 = float(x), float(x)
    while True:
        x1 = 0.5 * (x0 + c / x0)
        if abs(x1 - x0) < 1e-5:
            break
        x0 = x1
    return int(x1)


# print(mySqrt(99))

def longestPalindrome(s):
    n = len(s)
    if n < 2:
        return s

    dp = np.zeros((n, n))
    # dp = [[False for _ in range(n)] for _ in range(n)]
    # dp = [[False] * n] * n # 这么写有 bug ！！！
    for i in range(n):
        dp[i][i] = 1

    max_len = 1
    begin = 0
    for l in range(2, n + 1):
        for i in range(n):
            j = i + l - 1
            if j >= n:
                break

            if s[i] == s[j]:
                if j - i > 1:
                    dp[i][j] = dp[i + 1][j - 1]
                else:
                    dp[i][j] = 1
            else:
                dp[i][j] = 0
            # print(dp)
            if dp[i][j] == 1:
                max_len = l
                begin = i

    return s[begin:begin + max_len]


# print(longestPalindrome('saaasgds'))

def decodeString(s):
    # s = "3[a2[c]]"
    stack, res, multi = [], "", 0
    for i in s:
        if i == "[":
            stack.append((multi, res))
            multi, res = 0, ""
        elif i == "]":
            current_multi, last_res = stack.pop()
            res = last_res + current_multi * res
        elif "0" <= i <= "9":
            multi = multi * 10 + int(i)
        else:
            res += i
    return res


# print(decodeString("3[a2[c]]"))

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def sortList(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head
    # cut
    slow, fast = head, head.next
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    mid, slow.next = slow.next, None
    # merge
    left, right = sortList(head), sortList(mid)
    res = h = ListNode(0)
    while left and right:
        if left.val <= right.val:
            h.next, left = left, left.next
        else:
            h.next, right = right, right.next
        h = h.next
    h.next = left if left else right
    return res.next


# nums = np.random.randint(0, 10, (5,))
# print(nums)
#
# node = None
# n_node = None
# for i in nums:
#     if not node:
#         node = ListNode(i)
#     else:
#         n_node = ListNode(i)
#         n_node.next = node
#         node = n_node
#
# print(n_node.val, n_node.next.val)
#
# res = sortList(n_node)
# print(res.val, res.next)
# print(res.val, res.next.val)


def maxProfit(nums):
    min_price = float("inf")
    max_profit = -float("inf")
    for i in nums:
        min_price = min(i, min_price)
        max_profit = max(max_profit, i - min_price)
    return max_profit


# nums = np.random.randint(1, 10, (5,))
# print(nums)
# print(maxProfit(nums))

def merge(nums1, nums2, m, n):
    p1, p2 = m - 1, n - 1
    tail = m + n - 1
    while p1 >= 0 or p2 >= 0:
        if p1 == -1:
            nums1[tail] = nums2[p2]
            p2 -= 1
        elif p2 == -1:
            nums1[tail] = nums1[p1]
            p1 -= 1
        elif nums1[p1] > nums2[p2]:
            nums1[tail] = nums1[p1]
            p1 -= 1
        else:
            nums1[tail] = nums2[p2]
            p2 -= 1
        tail -= 1


# nums1 = [1, 3, 4, -1, -1]
# nums2 = [2, 6]
# merge(nums1, nums2, 3, 2)
# print(nums1)

def noRepeatLongestSubString(s):
    n = len(s)
    res = ""
    res_len = 0
    r = -1
    occur = set()
    for l in range(n):
        if l > 0:
            occur.remove(s[l - 1])
        while r + 1 < n and s[r + 1] not in occur:
            occur.add(s[r + 1])
            r += 1
        if r - l + 1 > res_len:
            res_len = r - l + 1
            res = s[l:r + 1]
    print(res)
    return res_len


# print(noRepeatLongestSubString("fdsfs"))


def calculate(s):
    sign, ops = 1, [1]

    i = 0
    n = len(s)
    ret = 0
    while i < n:
        _s = s[i]
        if _s == " ":
            i += 1
        elif _s == "+":
            sign = ops[-1]
            i += 1
        elif _s == "-":
            sign = -ops[-1]
            i += 1
        elif _s == "(":
            ops.append(sign)
            i += 1
        elif _s == ")":
            ops.pop()
            i += 1
        else:
            num = 0
            while i < n and s[i].isdigit():
                num = num * 10 + int(s[i])
                i += 1
            ret += num * sign
    return ret


# print(calculate("1 + (3 - 5)"))

def LCS(word1, word2):
    m, n = len(word1), len(word2)
    dp = np.zeros((m + 1, n + 1))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            # print(dp)
    word = ""
    i, j = m, n
    while i > 0 and j > 0:
        if word1[i - 1] == word2[j - 1]:
            word += word1[i - 1]
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return dp[m][n], word[::-1]


# print(LCS("fdsaord1", "word24434"))

def reorderList(head: ListNode) -> None:
    vec = []
    node = head
    while node:
        vec.append(node)
        node = node.next

    i, j = 0, len(vec) - 1
    while i < j:
        vec[i].next = vec[j]
        i += 1
        if i == j:
            break
        vec[j].next = vec[i]
        j -= 1
    vec[i].next = None


def dfs(grid, r, c):
    grid[r][c] = "0"
    nr, nc = len(grid), len(grid[0])
    for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
            dfs(grid, x, y)


def numIslands(grid):
    nr = len(grid)
    if nr == 0:
        return 0
    nc = len(grid[0])

    num_islands = 0
    for r in range(nr):
        for c in range(nc):
            if grid[r][c] == "1":
                num_islands += 1
                dfs(grid, r, c)
    return num_islands


def quickSelect(nums, k):
    l, m, r, p = [], [], [], nums[0]
    for n in nums:
        if n < p:
            l.append(n)
        elif n == p:
            m.append(n)
        else:
            r.append(n)

    if k <= len(r):
        return quickSelect(r, k)
    elif len(r) < k <= len(m + r):
        return m[0]
    else:
        return quickSelect(l, k - len(m + r))


def quickSelectPoint(nums, k):
    n = len(nums)
    l, r = 0, n - 1
    p = nums[l]
    while l < r:
        while (l < r) and (nums[r] >= p):
            r -= 1
        nums[l] = nums[r]
        while (l < r) and (nums[l] < p):
            l += 1
        nums[r] = nums[l]

    nums[l] = p
    if k <= n - l - 1:
        return quickSelectPoint(nums[l + 1:], k)
    elif k == n - l:
        return p
    else:
        return quickSelectPoint(nums[:l], k - n + l)


def searchRotated(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r + 1) // 2
        if nums[mid] == target:
            return mid
        if nums[l] < nums[mid]:
            if nums[l] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        else:
            if nums[mid] <= target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
    return -1


# nums = np.random.randint(0, 20, (10,)).tolist()
# print(sorted(nums))
# print(quickSelectPoint(nums, 3))

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


max_sum = float("-inf")


def maxPathSum(root: TreeNode) -> int:
    def maxGain(node: TreeNode) -> int:
        if not node:
            return 0

        left_gain = max(maxGain(node.left), 0)
        right_gain = max(maxGain(node.right), 0)

        new_path_sum = node.val + left_gain + right_gain

        max_sum = max(max_sum, new_path_sum)

        return node.val + max(left_gain, right_gain)

    maxGain(root)

    return max_sum


"""
# self_attention
q.shape = B,nh,T,hs
k.shape = B,nh,hs,T
att = q @ k # B,nh,T,T
v.shape = B,nh,T,hs
v = att @ v

"""

"""
# transformer
# emb
wte
wpe
drop1(wte + wpe)

# enc
x = emb(x)

_x = x
x = self_attention(q=x, k=x, v=x)
x = drop1(x)
x = ln1(x + _x)

_x = x
x = ffn(x)
x = drop2(x)
x = ln2(x + _x)

# dec
dec = emb(x)

_x = dec
x = self_attention(q=dec, k=dec, v=dec)
x = drop1(x)
x = ln1(x + _x)

_x = x
x = enc_dec_attention(q=x, k=enc, v=enc)
x = drop2(x)
x = ln2(x + _x)

_x = x
x = ffn(x)
x = drop3(x)
x = ln3(x + _x)

# lm_head

"""

"""
# lora
input x, columns
weight rows, columns
lora_right columns, lora_dim
lora_left lora_dim, rows

input @ lora_right @ lora_left

"""

"""
# bert
# emb
drop(wte + wpe + wse)

# enc
x = ln1(x)
x = self_attention(x)
x = x + drop1(x)

x = ln2(x)
x = ffn(x)
x = x + drop2(x)

"""

"""
# GPT2

# CasaulSelfAttention
q B,nh,T,hs
k B,nh,hs,T
v B,nh,T,hs
att = q @ k / math.sqrt(hs)
att = att.masked_fill(bias == 0, float("-inf"))
att = soft_max(att)
att = drop(att)
y = att @ v
y = y.transpose(1,2).view(B,T,C)
y = c_proj(y)
y = resid_drop(y)

# Block
x = x + attn(ln1(x))
x = x + mlpf(ln2(x))

# GPT
x = wte(x) + wpe(pos)
x = drop(x)
for x = block(x)
x = ln(x)
logits = lm_head(x)
loss = cross_entrophy(logits, targets)

"""

"""
# sgd with momentum
beta = 0.9
m_t = beta * m_t_old + (1 - beta) * g_t

# adam
beta1 = 0.9
m_t = beta1 * m_t_old + (1 - beta1) * g_t
beta2 = 0.99
v_t = beta2 * v_t_old + (1 - beta2) * g_t**2

eta = alpha * m_t / v_t**0.5

w_t_new = w_t - eta

"""

"""
f1 = 2 * acc * rec / (acc + rec)

acc = common / candi
recall = common / target

"""

"""
# dropout

mask = (torch.rand(x.shape) > dropout).float() 
x = mask * x / (1.0 - dropout)

"""

"""
sigmoid = 1 / (1 + exp(-z))

"""

"""
# lstm

f_t = sigmoid(w_f * cat(h_t_old, x_t) + b_f) # forget
i_t = sigmoid(w_i * cat(h_t_old, x_t) + b_i) # input
o_t = sigmoid(w_o * cat(h_t_old, x_t) + b_o) # output

c_h_t = tanh(w_c * cat(h_t_old, x_t) + b_c)

c_t = f_t * c_t_old + i_t * c_h_t

h_t = o_t * tanh(c_t)

"""

"""
# gru

r = sigmoid(w_r * cat(h_t_old, x_t)) # reset
z = sigmoid(w_z * cat(h_t_old, x_t)) # update

h_h_t_1 = h_t_1 * r
h_h = tanh(w * cat(h_h_t_1, x_t))
h_t = (1 - z) * h_t_1 + z * h_h

"""

"""
# llama
not ln, RMSNorm
SwiGLU
Rotary Embeddings：引入RoPE

"""

"""
# bloom
alibi

"""

"""
# muti_head

mul_64 = 64 * 1024 * 128 * 128
mul_1 = 1 * 1024 * 8192 * 8192

多头学到不同维度的东西
提升训练效率

注意力稀疏？

"""

"""

L1正则化使得权重w往0靠，使网络中的权重尽可能为0，也就相当于减小了网络复杂度，防止过拟合。
这也就是L1正则化会产生更稀疏（sparse）的解的原因。此处稀疏性指的是最优值中的一些参数为0。L1正则化的稀疏性质已经被广泛地应用于特征选择机制，从可用的特征子集中选择出有意义的特征。

因此在梯度下降过程中，权重 w 将逐渐减小，趋向于0但不等于0。这也就是权重衰减（weight decay）的由来。
L2正则化起到使得权重参数 w 变小的效果，为什么能防止过拟合呢？因为更小的权重参数 w 意味着模型的复杂度更低，对训练数据的拟合刚刚好，不会过分拟合训练数据，从而提高模型的泛化能力。

"""

"""
# tf-idf

x term, y document
tf = freq of x in y

df_x = num of doc include x
N = total num of doc
idf = log(N / df_x)
tf * idf

"""

"""
# fasttext

字符级 3-gram，有点类似 bpe

"""

"""

向量长度等于向量和自己的点积的开方

cos = np.dot(a, b) / (len(a) * len(b))

"""

"""
对比学习 simCLR
x = emb(x)
x = enc(x)
x = mlp(x)

infoNCE loss

loss = - log(exp(sim(x_i, x_j) / t) / sum(sim(x_i, x_j) / t))

"""

"""
CLIP

# image_encoder - Reset or Vision Transformer
# text_encoder - CBOW or Text Transformer
# IIn, h, w, cJ - minibatch of aligned images
# TIn, l] - minibatch of aligned texts
# W_ild_i, del- learned proj of image to embed
# W_tld_t, del - learned proj of text to embed
# t - learned temperature parameter
# extract feature representations of each modality

# 分别提取图像特征和文本特征
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# 对两个特征进行线性投射，得到相同维度的特征，并进行l2归一化
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# 计算缩放的余弦相似度：[n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# 对称的对比学习损失：等价于N个类别的cross_entropy_loss
labels = np.arange(n) # 对角线元素的labels
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2


"""

"""
softmax temperature

def softmax(vec, temperature):
    '''
    turn vec into normalized probability
    '''
    sum_exp = sum(math.exp(x/temperature) for x in vec)
    return [math.exp(x/temperature)/sum_exp for x in vec]

"""

"""
pytorch 训练流程

加载数据。 如果已完成本教程的上一步，则已经完成了数据加载。
1.定义模型。
2.定义损失函数。
使用训练数据训练模型。
使用测试数据测试网络。

# Loading the Data
df = pd.read_excel(r'C:…\Iris_dataset.xlsx') 
print('Take a look at sample from the dataset:') 
print(df.head()) 

# Let's verify if our data is balanced and what types of species we have  
print('\nOur dataset is balanced and has the following values to predict:') 
print(df['Iris_Type'].value_counts())

# Convert Iris species into numeric types: Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2.  
labels = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2} 
df['IrisType_num'] = df['Iris_Type']   # Create a new column "IrisType_num" 
df.IrisType_num = [labels[item] for item in df.IrisType_num]  # Convert the values to numeric ones 

# Define input and output datasets 
input = df.iloc[:, 1:-2]            # We drop the first column and the two last ones. 
print('\nInput values are:') 
print(input.head())   
output = df.loc[:, 'IrisType_num']   # Output Y is the last column  
print('\nThe output value is:') 
print(output.head())


# Convert Input and Output data to Tensors and create a TensorDataset 
input = torch.Tensor(input.to_numpy())      # Create tensor of type torch.float32 
print('\nInput format: ', input.shape, input.dtype)     # Input format: torch.Size([150, 4]) torch.float32 
output = torch.tensor(output.to_numpy())        # Create tensor type torch.int64  
print('Output format: ', output.shape, output.dtype)  # Output format: torch.Size([150]) torch.int64 
data = TensorDataset(input, output)    # Create a torch.utils.data.TensorDataset object for further data manipulation


# Split to Train, Validate and Test sets using random_split 
train_batch_size = 10        
number_rows = len(input)    # The size of our dataset or the number of rows in excel table.  
test_split = int(number_rows*0.3)  
validate_split = int(number_rows*0.2) 
train_split = number_rows - test_split - validate_split     
train_set, validate_set, test_set = random_split( 
    data, [train_split, validate_split, test_split])    
 
# Create Dataloader to read the data within batch sizes and put into memory. 
train_loader = DataLoader(train_set, batch_size = train_batch_size, shuffle = True) 
validate_loader = DataLoader(validate_set, batch_size = 1) 
test_loader = DataLoader(test_set, batch_size = 1)


# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Training Function 
def train(num_epochs): 
    best_accuracy = 0.0 
     
    print("Begin training...") 
    for epoch in range(1, num_epochs+1): 
        running_train_loss = 0.0 
        running_accuracy = 0.0 
        running_vall_loss = 0.0 
        total = 0 
 
        # Training Loop 
        for data in train_loader: 
        #for data in enumerate(train_loader, 0): 
            inputs, outputs = data  # get the input and real species as outputs; data is a list of [inputs, outputs] 
            optimizer.zero_grad()   # zero the parameter gradients          
            predicted_outputs = model(inputs)   # predict output from the model 
            train_loss = loss_fn(predicted_outputs, outputs)   # calculate loss for the predicted output  
            train_loss.backward()   # backpropagate the loss 
            optimizer.step()        # adjust parameters based on the calculated gradients 
            running_train_loss +=train_loss.item()  # track the loss value 
 
        # Calculate training loss value 
        train_loss_value = running_train_loss/len(train_loader) 
 
        # Validation Loop 
        with torch.no_grad(): 
            model.eval() 
            for data in validate_loader: 
               inputs, outputs = data 
               predicted_outputs = model(inputs) 
               val_loss = loss_fn(predicted_outputs, outputs) 
             
               # The label with the highest value will be our prediction 
               _, predicted = torch.max(predicted_outputs, 1) 
               running_vall_loss += val_loss.item()  
               total += outputs.size(0) 
               running_accuracy += (predicted == outputs).sum().item() 
 
        # Calculate validation loss value 
        val_loss_value = running_vall_loss/len(validate_loader) 
                
        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
        accuracy = (100 * running_accuracy / total)     
 
        # Save the model if the accuracy is the best 
        if accuracy > best_accuracy: 
            saveModel() 
            best_accuracy = accuracy 
         
        # Print the statistics of the epoch 
        print('Completed training batch', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Validation Loss is: %.4f' %val_loss_value, 'Accuracy is %d %%' % (accuracy))

'''
https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step

Recall that when initializing optimizer you explicitly tell it what parameters (tensors) of the model it should be updating. 
The gradients are "stored" by the tensors themselves (they have a grad and a requires_grad attributes) once you call backward() on the loss. 
After computing the gradients for all tensors in the model, calling optimizer.step() makes the optimizer iterate over all parameters (tensors) it is supposed to update and use their internally stored grad to update their values.
'''

"""

"""

bert 和 ELMo 区别
bert 用的 transformer ELMo 用的 LSTM

[MASK] 屏蔽 15% 的 token，只训练 [MASK] token
预训练和finetuning之间不匹配，因为在finetuning期间从未看到[MASK]token，未解决这个问题，有以下操作
80％的时间：用[MASK]标记替换单词，例如，my dog is hairy → my dog is [MASK]
10％的时间：用一个随机的单词替换该单词，例如，my dog is hairy → my dog is apple
10％的时间：保持单词不变，例如，my dog is hairy → my dog is hairy. 这样做的目的是将表示偏向于实际观察到的单词。

CLS]是用于分类输出的特殊符号，也是 next sentence 的训练 token

"""

"""
决策树
ID3 C4.5
ID3 
从信息论的知识中我们知道：信息熵越大，从而样本纯度越低。
ID3 算法的核心思想就是以信息增益来度量特征选择，选择信息增益最大的特征进行分裂。
算法采用自顶向下的贪婪搜索遍历可能的决策树空间（C4.5 也是贪婪搜索）。 其大致步骤为：
（1）初始化特征集合和数据集合；
（2）计算数据集合信息熵和所有特征的条件熵，选择信息增益最大的特征作为当前决策节点；
（3）更新数据集合和特征集合（删除上一步使用的特征，并按照特征值来划分不同分支的数据集合）；
（4）重复 2，3 两步，若子集值包含单一特征，则为分支叶子节点。


H(D) = -sum(k_p * log(k_p)) # H(D) 数据集合信息熵, k_p 第 k 类样本的概率

H(D∣A) = sum(p_a_i * H(p_a_i))
# H(D∣A) 读作 h (d given a) 针对某个特征 A，对于数据集 D 的条件熵 H(D∣A)
# p_a_i 特征 A 的取值 i 的概率, H(p_a_i) 特征 A 的取值 i 的数据信息熵

gain(D,A) = H(D) - H(D∣A) # 信息增益，为正，选 gain 最大的特征 A

sklearn
clf = tree.DecisionTreeClassifier
clf.fit(X,Y)
clf.feature_importances_

"""

"""
SVM
用超平面分割数据集

SVM（支持向量机）是一种监督学习方法，它主要用于分类和回归问题。在分类问题中，SVM试图找到一个最优的超平面（对于高维数据）或直线（对于二维数据），使得不同类别之间的间隔最大化。

假设我们有两个类别（类别A和类别B），并用一组有标签的数据表示它们。我们的目标是找到一个分隔它们的超平面，以确保正确的分类。

线性可分SVM

对于线性可分的情况，我们可以找到一个超平面（记作 w * x + b = 0）将两个类别完美分开。这里 w 是法向量，x 是数据点，b 是偏置。我们要最大化间隔 margin = 2 / ||w|| ，其中 margin 是正负样本到分割面的距离之和，||w|| 是w的范数（长度）。

要达到这个目标，我们需要解决以下优化问题：

minimize ||w||^2 / 2
subject to y_i (w * x_i + b) >= 1, i = 1, ..., N
其中 y_i 是数据点 x_i 的标签，N是总样本数。

线性不可分SVM

对于非线性可分的情况，我们引入松弛变量 ξ_i 来容忍错误分类。这需要我们调整优化问题如下：

minimize ||w||^2 / 2 + C * ∑ ξ_i
subject to y_i (w * x_i + b) >= 1 - ξ_i, i = 1, ..., N
ξ_i >= 0, i = 1, ..., N
参数 C 是一个非负系数，它决定了对错误分类的惩罚力度。较大的C会导致较小的间隔，以尽量减少错误。较小的C会产生较大的间隔，但可能允许更多的错误。

核技巧

如果数据线性不可分，我们可以通过核技巧将原始数据转换到一个更高维的空间，其中它们可能是线性可分的。这可以通过使用不同类型的核函数实现，如径向基函数（RBF）、多项式核等。核函数度量的是两个数据点在目标空间中的相似性。

求解优化问题

通过拉格朗日乘子法与对偶问题，我们可以将原始优化问题转换成只涉及核函数的优化问题。通常，我们采用序列最小优化（SMO）算法、梯度下降等方法来解决这个对偶问题。

如此，SVM便可找到一个最优的分隔超平面，以在给定任务中实现高准确率的分类。

"""

"""
PCA（主成分分析）是一种统计方法，用于降低数据维度，同时保留数据中最重要的信息。通过找到数据中方差最大的正交轴（称为主成分），我们可以减少数据的维数，同时保留尽可能多的信息。以下是PCA的数学原理：

中心化数据：为了消除数据中的偏移，首先我们需要计算数据的平均值（即每个特征的均值），然后用每个数据点减去对应的平均值。这样，新的数据集将以原点为中心。

计算协方差矩阵：接下来，我们要计算数据的协方差矩阵。协方差矩阵反映了特征之间的相关性。设X是一个维度为 m x n 的中心化数据矩阵（m个样本，n个特征），其中每一行表示一个观察样本，每一列表示一个特征。那么协方差矩阵C可通过以下公式计算：

C = (1 / (m - 1)) * (X^T * X)

其中，X^T 是X的转置矩阵，C为一个 nxn 的对称矩阵。

计算特征值和特征向量：在计算出协方差矩阵 C 之后，我们需要计算其特征值和对应的特征向量。特征向量指明了数据变换的方向，而特征值则表示这个方向上的方差大小。在PCA中，主成分对应于矩阵C的特征向量，方差对应于特征值。

排序主成分：将特征值按降序排列，并选择对应的特征向量。排序的目的是保证选取的主成分具有最大的方差。记特征向量组成的矩阵为 P（维度为 nxn）。

选择主成分和降维：我们可以选择前k个主成分（特征向量），其中k表示我们希望降低到的维度（k < n）。选择前k个特征向量组成的矩阵为P_k（维度为 nxk）。

投影数据：将原始数据矩阵X（维度为 m x n）投影到选择的k个主成分组成的子空间上，得到一个新的降维后的数据集 Y（维度为 m x k）：

Y = X * P_k

这样，我们就完成了用主成分分析对数据进行降维的过程。降维后的数据可以用于可视化、机器学习等后续任务。

"""

"""
GPT-2 Small：包含1.17亿（1,170,000,000）个参数
GPT-2 Medium：包含3.54亿（3,540,000,000）个参数
GPT-2 Large：包含7.74亿（7,740,000,000）个参数
GPT-2 Extra Large (GPT-2 XL)：包含15.5亿（15,500,000,000）个参数

"""
