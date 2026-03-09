/* Copyright 2026. Institute of Biomedical Imaging. TU Graz
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2026 Philip Schaten <philip.schaten@tugraz.at>
 */


console.log("Hello from bart_base")

// rpc

var request_pending = null;
var response_resolve = null;
var response_reject = null;

async function handle_msg(e)
{
    if (request_pending == null) {

        console.log("Received early/unsolicited msg. Please reload page!");
        return
    }

    data = e.data

    rx_data = data;

    if (0 == data[0])
        response_resolve(data[1])
    else
        response_reject(data[1])

    request_pending = null;
}

var rx_data;

// Promise allows 'blocking' until the response is received.
// further idea: Atomics.wait
// https://jasonformat.com/javascript-sleep/

function send_msg(msg)
{
    if (null != request_pending)
        return new Promise((resolve, reject) => {
            reject("request overlap")
        });

    request_pending = 1;

    return new Promise((resolve, reject) => {
        response_resolve = resolve;
        response_reject = reject;

        bart_worker.postMessage(msg);
    });
}

var bart_worker = new Worker("bart_worker.js")
bart_worker.onmessage = handle_msg;



// avoid letting exceptions and rejections through to python
// _pyodide/_future_helper.py", line 10, in set_exception
// TypeError: invalid exception object

async function py_wrapper(res)
{
    try {

        return [0, await res];
    } catch (e) {

        return [1, e];
    }
}

// api

function send_cfl(name)
{
    return py_wrapper(send_msg(['put_file', `${name}`, pyodide.FS.readFile(`${name}`)]));
}

function get_cfl(name)
{
    return py_wrapper(send_msg(['get_file', `${name}`]).then((hdr) => {
pyodide.FS.writeFile(`${name}`, hdr);
    }));
}

function rm_cfl(name)
{
    return py_wrapper(send_msg(['rm_file', `${name}`]));
}

function bart_cmd(cmd)
{
    return py_wrapper(send_msg(['bart_cmd', cmd]))
}

function reload_bart()
{
    return py_wrapper(send_msg(['reload_bart']))
}
