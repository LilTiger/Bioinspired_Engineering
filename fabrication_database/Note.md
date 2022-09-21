### 前端调用后端的方式
首先 使用如下语句引用axios插件 以提供必要的javascript工具  
`` <script src="https://unpkg.com/axios/dist/axios.min.js"></script>``

接下来的步骤分情况讨论：
- 如果生成图的 *javescript* 在该html文件中 使用``<script type="text/javascript"></script>``
语句并在中间写明具体调用内容
- 如果生成图的 *javescript* 在一个单独的 *js* 文件（如heart.html） 那么在该文件中写明调用（注意此时只需要写函数语句 不需要script前缀）

#### 具体的调用方式为(注意更改http网址)：
无需传参时:  
``
this.axios.get('http://www.dviz.cn/BiologicalStemCell/search/AllSecTopic')
        .then(res=>{
            // 接口数据
            console.log(res)
        })
    .catch(error=>{
        // 连接接口失败抛出错误
    })
``
需要传参时（器官智造数据库使用方法）:  
``
  let parameter = { param: '' }
            // param不传参是所有 传参的话是单个副主题的内容
    this.axios.get('http://www.dviz.cn/BiologicalStemCell/statistics/cell',
        { params: parameter})
        .then(res=>{
            // 接口数据
            console.log(res)
        })
    .catch(error=>{
        // 连接接口失败抛出错误
    })
``


打开网页 F12打开控制台-console 查看是否有data返回 状态是否200（否则联系后端）
使用 *res.data.xx* 调用具体数值 然后传到图标的label和data中即可


