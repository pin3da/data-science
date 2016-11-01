#! /usr/local/bin/node

var fs = require('fs')
var exec = require('child_process').exec

var name = process.argv[2]

fs.watch(name, function (eventType, filename) {
  if (filename) {
    var cmd = `pdflatex ${filename}`
    exec(cmd)
  }
})
