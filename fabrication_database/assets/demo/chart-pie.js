// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Interface Connection
    let parameter = { param: 'heart' }
axios.all([
    axios.get('http://www.dviz.cn/BiologicalStemCell/statistics/manufacturingStrategy',
        { params: parameter})
        .then(res=>{
            // 接口数据
            console.log(res)
            let label_strategy = res.data.data['axis'];
            let count_strategy = res.data.data['data'];

    // Pie Chart
    var ctx = document.getElementById("Pie-heart");
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
    axios.get('http://www.dviz.cn/BiologicalStemCell/statistics/cell',
        { params: parameter})
        .then(res=> {
          // 接口数据
          console.log(res)
          let label_strategy = res.data.data['axis'];
          let count_strategy = res.data.data['data'];
          // Bar Chart
          var ctx = document.getElementById("Bar-heart");
          var myLineChart = new Chart(ctx, {
            type: 'bar',
            data: {
              labels: ["BIOINK1", "BIOINK2", "BIOINK3", "BIOINK4", "BIOINK5", "BIOINK6"],
              datasets: [{
                label: "Revenue",
                backgroundColor: "rgba(2,117,216,1)",
                borderColor: "rgba(2,117,216,1)",
                data: [4215, 5312, 6251, 3496, 4987, 8795],
              }],
            },
            options: {
              scales: {
                xAxes: [{
                  time: {
                    unit: 'month'
                  },
                  gridLines: {
                    display: false
                  },
                  ticks: {
                    maxTicksLimit: 6
                  }
                }],
                yAxes: [{
                  ticks: {
                    min: 0,
                    max: 15000,
                    maxTicksLimit: 5
                  },
                  gridLines: {
                    display: true
                  }
                }],
              },
              legend: {
                display: false
              }
            }
          });
        })
             .catch(error=>{
            // 连接接口失败抛出错误
        })
            })
    ])

