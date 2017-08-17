var googleFinance = require('google-finance');
var fs = require("fs");
var json2csv = require("json2csv");
var dateformat = require("dateformat");

// var parse = require('csv-parse');
// var async = require('async');

const company = "AAPL";
const csvfilename = `./${company}.csv`;
const mvAvgIndex = 15;


googleFinance.historical({
        symbol: 'NASDAQ:' + company,
        from: '2007-04-01',
        to: '2017-04-01'
}, function (err, quotes) {
        // console.log(quotes);

        // var mvAvgValues = [];
        for(var i = 0; i < quotes.length; i++) {
                // quotes[i].date = dateformat(quotes[i], "yyyymmdd");
                var quote = quotes[i];
                var date = new Date(quote.date);
                quote.date = dateformat(date, "yyyymmdd");
                // quote["moving_avg"] = movingAverage(mvAvgValues, quote.close, mvAvgIndex, i);
        }
        var result = json2csv({data: quotes});
        fs.writeFile(csvfilename, result, "utf8", () => err ? console.log(err): console.log("success"))
});


function sum(arr) {
        return arr.reduce((a, b) => a + b, 0);
}

// function expMvAvg(closePrice, prevExpMvAvg, factor, mvAvgIndex, currentIndex) { 
//         return ((closePrice - prevExpMvAvg) * factor ) + prevExpMvAvg;
// }

// function movingAverage(currentMovingAvg, currentClose,  mvAvgIndex, currentIndex) {
//         var result = 0;
//         currentMovingAvg.push(currentClose);
//         if (currentIndex > mvAvgIndex - 1) {
//                 currentMovingAvg =  currentMovingAvg.slice(1);
//                 result = sum(currentMovingAvg) / mvAvgIndex;
//         }else if (i === mvAvgIndex - 1 ){
//                 result = sum(currentMovingAvg) / mvAvgIndex;
//         }else {
//                 result =  0;
//         }
//         return result;
// }


//                 if (i > moving_average_index - 1) {
//                         // moving_average_values = moving_average_values.slice(1)
//                         // moving_average_values.push(quote.close)
//                         // var current_moving_average = sum(moving_average_values) / moving_average_index;

//                         quote["moving_avg"] = movingAverage(moving_average_values, 
//                                                                 quote.close,
//                                                                 moving_average_index,
//                                                                 false);
//                 }else if (i === moving_average_index - 1 ){
//                         // moving_average_values.push(quote.close)
//                         // var current_moving_average = sum(moving_average_values) / moving_average_index;
//                         quote["moving_avg"] = movingAverage(moving_average_values, 
//                                                                 quote.close,
//                                                                 moving_average_index,
//                                                                 true);
//                 }else {
//                         moving_average_values.push(quote.close)
//                         quote["moving_avg"] = 0;
//                 }

