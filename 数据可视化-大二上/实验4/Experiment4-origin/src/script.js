import * as d3 from "d3";

let dataA = [
  { province: "Sichuan", students: 6, longitude: 104.06 },
  { province: "Hebei", students: 6, longitude: 114.48 },
  { province: "Yunnan", students: 1, longitude: 102.73 },
  { province: "Guizhou", students: 1, longitude: 106.71 },
  { province: "Hubei", students: 6, longitude: 114.31 },
  { province: "Henan", students: 13, longitude: 113.65 },
  { province: "Shandong", students: 9, longitude: 117.00 },
  { province: "Jiangsu", students: 5, longitude: 118.78 },
  { province: "Anhui", students: 3, longitude: 117.77 },
  { province: "Jiangxi", students: 5, longitude: 115.89 },
  { province: "Fujian", students: 5, longitude: 119.3 },
  { province: "Guangdong", students: 3, longitude: 113.23 },
  { province: "Hunan", students: 1, longitude: 113.00 },
  { province: "Liaoning", students: 4, longitude: 123.38 },
  { province: "Jilin", students: 1, longitude: 125.35 },
  { province: "Heilongjiang", students: 1, longitude: 126.63 },
  { province: "Shanxi", students: 2, longitude: 112.53 },
  { province: "Beijing", students: 1, longitude: 116.46 },
  { province: "Chongqing", students: 3, longitude: 106.54 },
  { province: "Tianjin", students: 15, longitude: 117.2 },
  { province: "Inner Mongolia", students: 6, longitude: 111.65 }
];

// 各省份的省会坐标[经度,纬度]
const geoCood = [
  { name: 'Gansu', geoCoord: [103.73, 36.03] },
  { name: 'Qinghai', geoCoord: [101.74, 36.56] },
  { name: 'Sichuan', geoCoord: [104.06, 30.67] },
  { name: 'Hebei', geoCoord: [114.48, 38.03] },
  { name: 'Yunnan', geoCoord: [102.73, 25.04] },
  { name: 'Guizhou', geoCoord: [106.71, 26.57] },
  { name: 'Hubei', geoCoord: [114.31, 30.52] },
  { name: 'Henan', geoCoord: [113.65, 34.76] },
  { name: 'Shandong', geoCoord: [117, 36.65] },
  { name: 'Jiangsu', geoCoord: [118.78, 32.04] },
  { name: 'Anhui', geoCoord: [117.27, 31.86] },
  { name: 'Zhejiang', geoCoord: [120.19, 30.26] },
  { name: 'Jiangxi', geoCoord: [115.89, 28.68] },
  { name: 'Fujian', geoCoord: [119.3, 26.08] },
  { name: 'Guangdong', geoCoord: [113.23, 23.16] },
  { name: 'Hunan', geoCoord: [113, 28.21] },
  { name: 'Hainan', geoCoord: [110.35, 20.02] },
  { name: 'Liaoning', geoCoord: [123.38, 41.8] },
  { name: 'Jilin', geoCoord: [125.35, 43.88] },
  { name: 'Heilongjiang', geoCoord: [126.63, 45.75] },
  { name: 'Shanxi', geoCoord: [112.53, 37.87] },
  { name: 'Shaanxi', geoCoord: [108.95, 34.27] },
  { name: 'Taiwan', geoCoord: [121.30, 25.03] },
  { name: 'Beijing', geoCoord: [116.46, 39.92] },
  { name: 'Shanghai', geoCoord: [121.48, 31.22] },
  { name: 'Chongqing', geoCoord: [106.54, 29.59] },
  { name: 'Tianjin', geoCoord: [117.2, 39.13] },
  { name: 'Inner Mongolia', geoCoord: [111.65, 40.82] },
  { name: 'Guangxi', geoCoord: [108.33, 22.84] },
  { name: 'Tibet', geoCoord: [91.11, 29.97] },
  { name: 'Ningxia', geoCoord: [106.27, 38.47] },
  { name: 'Xinjiang', geoCoord: [87.68, 43.77] },
  { name: 'Hong Kong', geoCoord: [114.17, 22.28] },
  { name: 'Macau', geoCoord: [113.54, 22.19] }
];

// 将缺失的省份添加到 dataA 数组中
geoCood.forEach(province => {
  if (!dataA.some(d => d.province === province.name)) {
    dataA.push({
      province: province.name,
      students: 0,
      longitude: province.geoCoord[0]
    });
  }
});

// // 手动聚合纬度较低的省份
// const lowLatitudeProvinces = dataA.filter(d => d.longitude < 100);
// const aggregatedLowLatitude = {
//   province: lowLatitudeProvinces.map(d => d.province).join(", "),
//   students: d3.sum(lowLatitudeProvinces, d => d.students),
//   longitude: `${d3.min(lowLatitudeProvinces, d => d.longitude).toFixed(2)} - ${d3.max(lowLatitudeProvinces, d => d.longitude).toFixed(2)}`
// };

// // 移除聚合的省份并添加聚合后的数据
dataA = dataA.filter(d => d.longitude >= 100);
// dataA.push(aggregatedLowLatitude);


const tianjin_longitude = 117.2;
const maxStudents = d3.max(dataA, d => d.students);
const totalStudents = d3.sum(dataA, d => d.students);

// Chart dimensions
const width = 900;
const height = 2000;

// Create scale for positioning
const yScaleA = d3.scaleLinear()
  .domain(d3.extent(dataA, d => d.longitude))
  .range([height - 100, 100]);

// Tianjin's x-coordinate
const tianjinX = width - 50;

// Create SVG
const svgA = d3.select("#chartA").append("svg")
  .attr("width", width)
  .attr("height", height);



// Label for province with the highest student count
const maxProvince = dataA.find(d => d.students === maxStudents);
svgA.append("text")
  .attr("x", width - 400)
  .attr("y", 120)
  .attr("dy", "0.35em")
  .style("font-size", "20px")
  .style("font-weight", "bold")
  .text(`Top Province: ${maxProvince.province} (${maxProvince.students} students)`);

// Statistics on student distribution by longitude
svgA.append("text")
  .attr("x", 210)
  .attr("y", yScaleA(maxProvince.longitude) / 2)
  .attr("dy", "0.35em")
  .style("font-size", "30px")
  .style("fill", "blue")
  .text(`19.59% from provinces east of Tianjin`);

svgA.append("text")
  .attr("x", 210)
  .attr("y", height - (height - yScaleA(maxProvince.longitude)) / 2)
  .attr("dy", "0.35em")
  .style("font-size", "30px")
  .style("fill", "red")
  .text(`64.95% from provinces west of Tianjin`);

// Additional interval labels for specific ranges
svgA.append("text")
  .attr("x", width - 400)
  .attr("y", 70)
  .attr("dy", "0.35em")
  .style("font-size", "20px")
  .style("font-weight", "bold")
  .text("Most common longitude range: 113 - 117");

svgA.append("text")
  .attr("x", width - 400)
  .attr("y", 95)
  .attr("dy", "0.35em")
  .style("font-size", "20px")
  .style("font-weight", "bold")
  .text("Farthest student origin: Heilongjiang (126.63)");

// Add total student count
svgA.append("text")
  .attr("x", width - 400)
  .attr("y", 50)
  .style("font-size", "20px")
  .style("font-weight", "bold")
  .text(`Total Students (excluding overseas): ${totalStudents}`);



// 定义线性渐变
const defs = svgA.append("defs");

// 定义蓝色渐变
const blueGradient = defs.append("linearGradient")
  .attr("id", "blue-gradient")
  .attr("x1", "0%")
  .attr("y1", "0%")
  .attr("x2", "100%")
  .attr("y2", "0%");

blueGradient.append("stop")
  .attr("offset", "0%")
  .attr("stop-color", "rgba(70, 130, 180, 0)") // 起始颜色，透明
  .attr("stop-opacity", 0);

blueGradient.append("stop")
  .attr("offset", "50%")
  .attr("stop-color", "rgba(70, 130, 180, 1)") // 中间颜色，不透明
  .attr("stop-opacity", 1);

blueGradient.append("stop")
  .attr("offset", "100%")
  .attr("stop-color", "rgba(70, 130, 180, 0)") // 结束颜色，透明
  .attr("stop-opacity", 0);

// 定义红色渐变
const redGradient = defs.append("linearGradient")
  .attr("id", "red-gradient")
  .attr("x1", "0%")
  .attr("y1", "0%")
  .attr("x2", "100%")
  .attr("y2", "0%");

redGradient.append("stop")
  .attr("offset", "0%")
  .attr("stop-color", "rgba(255, 0, 0, 0)") // 起始颜色，透明
  .attr("stop-opacity", 0);

redGradient.append("stop")
  .attr("offset", "50%")
  .attr("stop-color", "rgba(255, 0, 0, 1)") // 中间颜色，不透明
  .attr("stop-opacity", 1);

redGradient.append("stop")
  .attr("offset", "100%")
  .attr("stop-color", "rgba(255, 0, 0, 0)") // 结束颜色，透明
  .attr("stop-opacity", 0);

// 定义灰色渐变
const grayGradient = defs.append("linearGradient")
  .attr("id", "gray-gradient")
  .attr("x1", "0%")
  .attr("y1", "0%")
  .attr("x2", "100%")
  .attr("y2", "0%");

grayGradient.append("stop")
  .attr("offset", "0%")
  .attr("stop-color", "rgba(0, 0, 0, 0)") // 起始颜色，透明
  .attr("stop-opacity", 0);

grayGradient.append("stop")
  .attr("offset", "50%")
  .attr("stop-color", "rgba(0, 0, 0, 1)") // 中间颜色，不透明
  .attr("stop-opacity", 1);

grayGradient.append("stop")
  .attr("offset", "100%")
  .attr("stop-color", "rgba(0, 0, 0, 0)") // 结束颜色，透明
  .attr("stop-opacity", 0);

// 绘制从每个省到天津的曲线
dataA.forEach(d => {
  const startX = 200;
  const startY = yScaleA(d.longitude);
  const endY = yScaleA(tianjin_longitude)+0.001; 

  const distance = Math.abs(startY - endY);
  const curvatureFactor = 0.3 * (distance / height); 
  const controlPointX = (startX + tianjinX) / 2;
  const controlPointY = startY + (endY - startY) * curvatureFactor;

  const line = d3.path();
  line.moveTo(startX, startY);
  line.quadraticCurveTo(controlPointX, controlPointY, tianjinX, endY);

  // 定义颜色和宽度
  let gradientId;
  let strokeWidth;

  if (d.province === "Tianjin") {
    gradientId = "gray-gradient"; // 天津的灰色渐变
    strokeWidth = d.students;
  } else if (d.longitude >= tianjin_longitude) {
    gradientId = "blue-gradient"; // 天津以东省份的蓝色渐变
    strokeWidth = d.students;
  } else {
    gradientId = "red-gradient"; // 天津以西省份的红色渐变
    strokeWidth = d.students;
  }

  svgA.append("path")
    .attr("d", line.toString())
    .attr("fill", "none")
    .attr("stroke", `url(#${gradientId})`) // 使用对应的渐变
    .attr("stroke-width", strokeWidth)
    .attr("stroke-linecap", "round");
});

svgA.selectAll(".province-label")
  .data(dataA)
  .enter()
  .append("text")
  .filter(d => d.students !== 0)
  .attr("class", "province-label")
  .attr("x", 210)
  .attr("y", d => yScaleA(d.longitude))
  .attr("dy", "0.35em")
  .style("font-size", "12px")
  .text(d => d.province);

// Draw Tianjin location
svgA.append("circle")
  .attr("cx", tianjinX)
  .attr("cy", yScaleA(tianjin_longitude))
  .attr("r", 8)
  .attr("fill", "black")
  .attr("stroke", "black")
  .attr("stroke-width", 1.5);

svgA.append("text")
  .attr("x", tianjinX + 8)
  .attr("y", yScaleA(tianjin_longitude))
  .attr("dy", "0.35em")
  .style("font-size", "15px")
  .text("Tianjin");

const sortedDataA = dataA.slice().sort((a, b) => b.longitude - a.longitude);

// Define margins and dimensions for the distribution chart
const margin = { top: 20, right: 20, bottom: 20, left: 100 };
const chartWidth = 190; // Width of the left-side distribution chart
const distributionX = d3.scaleLinear()
  .domain([0, d3.max(dataA, d => d.students)])
  .range([chartWidth, 0]); // Invert the scale to extend left

// Add an inverted x-axis for the distribution chart at the bottom, facing left
svgA.append("g")
  .attr("transform", `translate(0, ${height - margin.bottom})`)
  .call(d3.axisBottom(distributionX).ticks(5)) // Inverted axis to go from right to left
  .selectAll("text")
  .style("font-size", "10px");

// Add labels for the regions of distribution, such as "higher students count" or "lower students count"
svgA.append("text")
  .attr("x", 10) // Position closer to the left end of the area chart
  .attr("y", yScaleA(d3.min(dataA, d => d.longitude)) - 50)
  .style("font-size", "20px")
  .attr("fill", "red")
  .text("West");

svgA.append("text")
  .attr("x", 10)
  .attr("y", yScaleA(d3.max(dataA, d => d.longitude)) + 50)
  .style("font-size", "20px")
  .attr("fill", "blue")
  .text("East");

// 定义线性渐变
const defs2 = svgA.append("defs");

const gradient = defs2.append("linearGradient")
  .attr("id", "blue-to-red-gradient")
  .attr("x1", "0%")
  .attr("y1", "0%")
  .attr("x2", "0%")
  .attr("y2", "100%");

gradient.append("stop")
  .attr("offset", "0%")
  .attr("stop-color", "steelblue")
  .attr("stop-opacity", 0.6);

gradient.append("stop")
  .attr("offset", "100%")
  .attr("stop-color", "red")
  .attr("stop-opacity", 0.6);

// 创建区域生成器
const areaGenerator = d3.area()
  .x0(chartWidth) // 起始 x 位置
  .x1(d => distributionX(d.students)) // 将学生数量映射到 x 位置
  .y(d => yScaleA(d.longitude))
  .curve(d3.curveBasis); // 平滑曲线

// 绘制区域路径
svgA.append("path")
  .datum(sortedDataA) // 使用排序后的数据
  .attr("d", areaGenerator)
  .attr("fill", "url(#blue-to-red-gradient)") // 使用线性渐变
  .attr("opacity", 0.6)
  .attr("stroke",  "url(#blue-to-red-gradient)");


/** ------------------ Chart B: Optimized Chart from Experiment 3 ------------------ */

// Data for Chart B (extracted from nobel_final csv)
const dataB = [
  { category: "[0, 20)", value: 1 },
  { category: "[20, 30)", value: 2 },
  { category: "[30, 40)", value: 57 },
  { category: "[40, 50)", value: 170 },
  { category: "[50, 60)", value: 243 },
  { category: "[60, 70)", value: 251 },
  { category: "[70, 80)", value: 158 },
  { category: "[80, 90)", value: 39 },
  { category: "[90, 100]", value: 2 }
];

// 设置 SVG 宽度和高度
const chartWidth2 = 900; // 定义更小的宽度来让它居中
const chartHeight = 800;

const svgB = d3.select("#chartB").append("svg")
  .attr("width", chartWidth2)
  .attr("height", chartHeight)
  .style("display", "block")  // 使用块级元素
  .style("margin", "0 auto"); // 居中显示

// 为 Chart B 创建比例尺
const xScaleB = d3.scaleBand()
  .domain(dataB.map(d => d.category))
  .range([50, chartWidth2 - 50])
  .padding(0.2);

const yScaleB = d3.scaleLinear()
  .domain([0, d3.max(dataB, d => d.value)])
  .range([chartHeight - 100, 50]);

const colorScale = d3.scaleSequential()
  .domain([0, d3.max(dataB, d => d.value)])
  .interpolator(d3.interpolateBlues);

// 为 Chart B 绘制柱状图
svgB.selectAll(".bar")
  .data(dataB)
  .enter()
  .append("rect")
  .attr("class", "bar")
  .attr("x", d => xScaleB(d.category))
  .attr("y", d => yScaleB(d.value))
  .attr("width", xScaleB.bandwidth())
  .attr("height", d => chartHeight - 100 - yScaleB(d.value))
  .attr("fill", d => colorScale(d.value))
  .attr("opacity", 0.7);

// 为 Chart B 添加柱状图标签
svgB.selectAll(".bar-label")
  .data(dataB)
  .enter()
  .append("text")
  .attr("class", "bar-label")
  .attr("x", d => xScaleB(d.category) + xScaleB.bandwidth() / 2)
  .attr("y", d => yScaleB(d.value) - 5)
  .attr("text-anchor", "middle")
  .style("font-size", "12px")
  .text(d => d.value);

// 为 Chart B 添加 x 轴
svgB.append("g")
  .attr("transform", `translate(0,${chartHeight - 100})`)
  .call(d3.axisBottom(xScaleB));

// 为 Chart B 添加 y 轴
svgB.append("g")
  .attr("transform", `translate(50,0)`)
  .call(d3.axisLeft(yScaleB));

// Add a trend line (smooth line) based on the data points
const lineGenerator = d3.line()
  .x(d => xScaleB(d.category) + xScaleB.bandwidth() / 2)
  .y(d => yScaleB(d.value))
  .curve(d3.curveMonotoneX); // Smooth curve

svgB.append("path")
  .datum(dataB)
  .attr("fill", "none")
  .attr("stroke", "blue")
  .attr("stroke-width", 2)
  .attr("d", lineGenerator);

// 在右上角添加 Min age 和 Max age 信息
const minAge = 17; // 最小年龄
const maxAge = 97; // 最大年龄

svgB.append("text")
  .attr("x", chartWidth + 600)
  .attr("y", 20)
  .attr("text-anchor", "end")
  .style("font-size", "20px")
  .style("font-weight", "bold")
  .text(`Min age: ${minAge}`);

svgB.append("text")
  .attr("x", chartWidth + 600)
  .attr("y", 40)
  .attr("text-anchor", "end")
  .style("font-size", "20px")
  .style("font-weight", "bold")
  .text(`Max age: ${maxAge}`);