const path = require('path')
const HtmlWebpackPlugin = require('html-webpack-plugin')
const CopyWebpackPlugin = require('copy-webpack-plugin')

async function getConfig() {
	return {
		entry: {
			home: './src/index.js'
		},
		output: {
			path: path.resolve(__dirname, 'dist')
		},
		module: {
			rules: [
				{
					test: /\.(ttf|woff|eot)$/,
					loader: 'file-loader',
					options: {
						name: '[name].[ext]?[hash]'
					}
				},
				{
					test: /\.(sa|sc|c)ss$/,
					use: ['style-loader', 'css-loader', 'sass-loader']
				}
			]
		},
		plugins: [
			new HtmlWebpackPlugin({template: 'src/index.html'}),
			new CopyWebpackPlugin({patterns: [{
				from: 'src/assets',
				to: 'assets'
			}]}),
		],
		devServer: {
			contentBase: path.join(__dirname, 'dist'),
			compress: true,
			port: 9000
		}
	}
}

module.exports = getConfig()