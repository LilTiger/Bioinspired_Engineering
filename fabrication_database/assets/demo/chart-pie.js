// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

    let parameter = { param: 'heart' }
    axios.get('http://www.dviz.cn/BiologicalStemCell/statistics/manufacturingStrategy',
        { params: parameter})
        .then(res=>{
            // 接口数据
            console.log(res)
            let label_strategy = res.data.data['axis'];
            let count_strategy = res.data.data['data'];

    // Pie Chart
    var ctx = document.getElementById("myPieChart");
    var myPieChart = new Chart(ctx, {
      type: 'pie',
      data: {
        labels: label_strategy,
        datasets: [{
          data: count_strategy,
          backgroundColor: ['#007bff', '#dc3545', '#ffc107', '#28a745'],
        }],
      },
    })
        .catch(error=>{
            // 连接接口失败抛出错误
        })
            })